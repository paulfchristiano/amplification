import tensorflow as tf

from amplification.models.answerer import AnswererWithTarget
from amplification.models.core import flatten_first, multinomial
from amplification import tf_utils

class TokenTypes():
    subQ = 0
    subA = 1
    mainQ = 2
    mainA = 3
    num = 4

def ints(shape, x):
    return x * tf.ones(shape, dtype=tf.int32)

def ints_like(t, x):
    return ints(tf.shape(t), x)

class AttentionSequenceModel(tf_utils.Model):
    def set_params(self, task, nh=128, nheads=4, depth=4, p_drop=0.1):
        self.nvocab = task.nvocab
        self.max_length = task.transcript_length
        self.nheads = nheads
        self.p_drop = p_drop
        self.nh = nh
        self.depth = depth
        self.alength = task.answer_length

    def encode(self, ws, token_types, is_training, start_position=0, context=None):
        #XXX batch norm here?
        if context is None: context = self.context["embed"]
        word_embeddings = tf_utils.embed(ws, self.nvocab, self.nh, context=context["words"])
        token_type_embeddings = tf_utils.embed(token_types, TokenTypes.num, self.nh, context=context["token_type"])
        nbatch = tf.shape(ws)[0]
        nw = tf.shape(ws)[1]
        positions = tf.expand_dims(tf.range(start_position, nw + start_position), 0)
        position_embeddings = tf_utils.embed(positions, self.max_length + 1, self.nh, context=context["position"])
        result = word_embeddings + token_type_embeddings + position_embeddings
        return tf_utils.dropout(result, p_drop=self.p_drop, do_dropout=is_training)

    def run(self, ws, token_types, context=None, prev_ys=None, is_training=tf.constant(False)):
        if context is None: context = self.context["run"]
        nbatch = tf.shape(ws)[0]
        if prev_ys is None:
            ws = tf.concat([ints((nbatch, 1), 0), ws], axis=1) #prepend ? as a start token
            token_types = tf.concat([ints((nbatch, 1), TokenTypes.mainQ), token_types], axis=1)
        start_position = tf.constant(0) if prev_ys is None else tf.shape(prev_ys[0])[1]
        inputs = self.encode(ws, token_types, start_position=start_position, context=context["encode"], is_training=is_training)
        new_ys = [inputs]
        cells = context.sequential(inputs)
        for i in range(self.depth):
            y = tf.concat([prev_ys[i], cells.value], axis=1) if prev_ys else cells.value
            cells.transformer_cell(self.nheads, y, training=is_training, p_drop=self.p_drop,
                    attention_args={'mask':True, 'mask_offset':start_position})
            new_ys.append(cells.value)
        cells = cells.value

        Q = tf_utils.fully_connected(cells, (self.nh,), context=context["Q"])
        K = tf_utils.fully_connected(cells, (self.nh,), context=context["K"])
        onehot_ws = tf.one_hot(ws, depth=self.nvocab)
        new_ys.append(K)
        new_ys.append(onehot_ws)
        if prev_ys is not None:
            K = tf.concat([prev_ys[-2], K], axis=1)
            onehot_ws = tf.concat([prev_ys[-1], onehot_ws], axis=1)
        w_logits = (onehot_ws - 1) * 1e9
        copy_logits = tf_utils.attention(Q, K, w_logits, True, mask_offset=start_position, logits=True)
        should_copy = (context.sequential(cells, name="should_copy").
                fully_connected((self.nh,)).relu().
                fully_connected(())).value
        output_logits = (context.sequential(cells, name="output").
                fully_connected((self.nh,)).relu().
                fully_connected((self.nvocab,))).value
        logits = tf_utils.soft_combine(copy_logits, output_logits, should_copy)
        logits, next_logits = logits[:, :-1], logits[:, -1]
        return logits, next_logits, new_ys

    def sample_next(self, ws, token_types, length=1, context=None):
        _, next_logits, ys = self.run(ws, token_types=token_types, context=context)
        def append_to_ys(ys, new_ys): return [tf.concat([y, new_y], axis=1) for y, new_y in zip(ys, new_ys)]
        ws = []
        all_logits = []
        for j in range(length):
            w = multinomial(next_logits)
            w = tf.expand_dims(w, -1)
            ws.append(w)
            all_logits.append(next_logits)
            if j < length:
                _, next_logits, new_ys = self.run(w, token_types=ints_like(w, TokenTypes.subQ), context=context, prev_ys=ys)
                ys = append_to_ys(ys, new_ys)
        return tf.concat(ws, axis=1), tf.stack(all_logits, axis=1)

    def sample(self, Q, rounds, answerer, context=None):
        qlength = int(Q.shape[1])
        if context is None: context = self.context["run"]
        _, next_logits, ys = self.run(Q, token_types=ints_like(Q, TokenTypes.mainQ), context=context)
        def append_to_ys(ys, new_ys): return [tf.concat([y, new_y], axis=1) for y, new_y in zip(ys, new_ys)]
        subQs = []
        subAs = []
        for i in range(rounds):
            ws = []
            for j in range(qlength):
                w = multinomial(next_logits) #happens to be expanded already...
                w = tf.expand_dims(w, -1)
                ws.append(w)
                _, next_logits, new_ys = self.run(w, token_types=ints_like(w, TokenTypes.subQ), context=context, prev_ys=ys)
                ys = append_to_ys(ys, new_ys)
            subQ = tf.concat(ws, axis=1)
            subQs.append(subQ)
            subA = answerer(subQ)
            subAs.append(subA)
            _, next_logits, new_ys = self.run(subA, token_types=ints_like(subA, TokenTypes.subA), context=context, prev_ys=ys)
            ys = append_to_ys(ys, new_ys)
        ws = []
        for i in range(self.alength):
            w = multinomial(next_logits)
            w = tf.expand_dims(w, -1)
            ws.append(w)
            _, next_logits, new_ys = self.run(w, token_types=ints_like(w, TokenTypes.mainA), context=context, prev_ys=ys)
            ys = append_to_ys(ys, new_ys)
        A = tf.concat(ws, axis=1)
        subQs = tf.stack(subQs, axis=1)
        subAs = tf.stack(subAs, axis=1)
        return subQs, subAs, A

    def build(self, ws, token_types, is_training=tf.constant(True)):
        logits, _, _ = self.run(ws, token_types, is_training=is_training)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ws)
        is_question = tf.cast(tf.equal(token_types, TokenTypes.subQ), dtype=tf.float32)
        is_answer = tf.cast(tf.equal(token_types, TokenTypes.mainA), dtype=tf.float32)
        Q_loss = tf.reduce_sum(losses * is_question) / tf.reduce_sum(is_question)
        A_loss = tf.reduce_sum(losses * is_answer) / tf.reduce_sum(is_answer)
        loss = 0.5 * (Q_loss + A_loss)
        train_op =  tf_utils.minimize(loss, self.context,
                clip_grads=1.0,
                context=self.context["train"],
                optimizer_args=dict(learning_rate=1e-5, beta2=0.98))
        return {"losses":losses, "loss":loss, "train":train_op}

class AskerAndAnswerer(tf_utils.Model):
    def set_params(self, task, answerer={}, asker={}, joint={}):
        answerer_args = {}
        asker_args = {}
        self.rounds = task.interaction_length
        joint["task"] = task
        answerer_args.update(answerer)
        answerer_args.update(joint)
        asker_args.update(asker)
        asker_args.update(joint)
        self.asker = AttentionSequenceModel(**asker_args, context=self.context["asker"])
        self.answerer = AnswererWithTarget(**answerer_args, context=self.context["answerer"])

    def asker_device(self):
        return tf.device("/device:GPU:0")

    def build(self, facts, Qs, targets, transcripts, token_types, is_training, simple_answerer):
        answerer_ops = self.answerer.build(facts=facts, Qs=Qs, targets=targets, is_training=is_training)
        flat_Qs, unflatten = flatten_first(Qs)
        answerer = self.answerer.answer_fn(facts, simple_answerer, unflatten=unflatten)
        batch_answerer = self.answerer.answer_fn(facts, simple_answerer)
        subQs, subAs, As = self.asker.sample(flat_Qs, self.rounds, answerer) 
        with self.asker_device():
            asker_ops = self.asker.build(ws=transcripts, token_types=token_types,
                    is_training=is_training)
        return {"answerer":answerer_ops,
                "asker":asker_ops,
                "targets":unflatten(As),
                "teacher_or_simple":batch_answerer(Qs),
                "subQs":unflatten(subQs),
                "subAs":unflatten(subAs)}

def make_transcript(main_Q, sub_Qs, sub_As, main_A=None):
    transcript = []
    tokens = []
    transcript.extend(main_Q)
    tokens.extend([TokenTypes.mainQ] * len(main_Q))
    for sub_Q, sub_A in zip(sub_Qs, sub_As):
        transcript.extend(sub_Q)
        tokens.extend([TokenTypes.subQ] * len(sub_Q))
        transcript.extend(sub_A)
        tokens.extend([TokenTypes.subA] * len(sub_A))
    if main_A is not None:
        transcript.extend(main_A)
        tokens.extend([TokenTypes.mainA] * len(main_A))
    return transcript, tokens
