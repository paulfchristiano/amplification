import tensorflow as tf

import amplification.tf_utils as tf_utils
from amplification.models.attention import AttentionModel
from amplification.models.core import flatten_first, flatten_to


class AttentionAnswerer(AttentionModel):
    def build(self, facts, Qs, targets, is_training=tf.constant(True)):
        state = self.run(facts, is_training)
        As, losses = self.answer(state, facts, Qs, is_training=is_training, targets=targets)
        loss = tf.reduce_mean(losses)
        def make_train():
            return tf_utils.minimize(loss, self.context, clip_grads=1.0, context=self.context["train"],
                optimizer_args=dict(learning_rate=1e-5, beta2=0.98))
        train_op = tf.cond(is_training, make_train, lambda : tf.no_op())
        return {"loss": loss, "train": train_op, "As":As, 'losses': losses}


class AnswererWithTarget(tf_utils.Model):
    def set_params(self, ema_horizon=1000, model_class=AttentionAnswerer, **kwargs):
        self.student = model_class(context=self.context["student"], **kwargs)
        self.teacher = model_class(context=self.context["teacher"], **kwargs)
        self.horizon = ema_horizon

    def student_device(self):
        return tf.device("/device:GPU:1")
    
    def teacher_device(self):
        return tf.device("/device:GPU:0")

    def update_ema_op(self):
        context = self.context["ema"]
        maintain_ema_ops = tf_utils.moving_averages(
                self.context["student"].map(filter = lambda x, t : "trainable" in t or "stats" in t),
                horizon=context.counter("horizon", cap=self.horizon),
                store_averages_in = self.context["teacher"],
            )
        return tf.group(*maintain_ema_ops)

    def build(self, facts, Qs, targets, is_training=tf.constant(True)):
        with self.student_device():
            student_ops = self.student.build(facts, Qs, targets, is_training=is_training)
        with self.teacher_device():
            teacher_ops = self.teacher.build(facts, Qs, targets, is_training=tf.constant(False))
        ops = {"student":student_ops, "teacher":teacher_ops}
        with tf.control_dependencies([student_ops["train"]]):
            update_ema = self.update_ema_op()
        ops["train"] = update_ema
        return ops

    def answer_fn(self, facts, simple_answerer, unflatten=None):
        with self.teacher_device():
            state = self.teacher.run(facts, tf.constant(False))
        def answer(Qs):
            if unflatten is not None:
                Qs = unflatten(Qs)
            is_simple, simple_As = simple_answerer(Qs)
            with self.teacher_device():
                complex_As = self.teacher.answer(state, facts, Qs, is_training=tf.constant(False))
            #tf.where doesn't broadcast, so...
            is_simple = tf.expand_dims(is_simple, -1)
            is_simple = tf.logical_or(is_simple, tf.zeros(tf.shape(simple_As), tf.bool)) 
            As = tf.where(is_simple, simple_As, complex_As)
            if unflatten is not None:
                As, _ = flatten_first(As)
            return As
        return answer
