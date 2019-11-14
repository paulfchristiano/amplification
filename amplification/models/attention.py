import tensorflow as tf

import amplification.tf_utils as tf_utils
from amplification.models.rnn import RNNModel
from amplification.models.core import flatten_first, flatten_to, multinomial


class AttentionModel(tf_utils.Model):
    def set_params(self, task, nh=512, ne=100, nheads=8, depth=6, answer_depth=3, scale_weights=True, p_drop=0.1):
        self.nvocab = task.nvocab
        self.nheads = nheads
        self.ne = ne
        self.nh = nh
        self.depth = depth
        self.answer_depth = answer_depth
        self.scale_weights = scale_weights
        self.p_drop = p_drop
        self.alength = task.answer_length

    def encode(self, ws, context=None):
        if context is None: context = self.context["encode"]
        encoding = tf_utils.embed(ws, self.nvocab, self.ne, context=context["embed"])
        return tf_utils.fully_connected(encoding, (self.nh,), in_axes=2, context=context["fc"])

    def encode_many(self, ws, is_training, context=None, batchnorm=True):
        if context is None: context = self.context["encode"]
        flat_ws, unflatten = flatten_first(ws)
        flat_result = self.encode(flat_ws, context=context["encode_one"])
        if batchnorm:
            flat_result = tf_utils.batch_norm(flat_result, context=context["bn"], training=is_training)
        return unflatten(flat_result)

    def run(self, ws, is_training, context=None):
        if context is None: context = self.context["run"]
        embedded_ws = self.encode_many(ws, is_training=is_training, context=context["encode"])
        state = context.sequential(embedded_ws)
        for i in range(self.depth):
            state.transformer_cell(self.nheads, scale_weights=self.scale_weights, training=is_training, p_drop=self.p_drop)
        state = state.value
        return state

    def generate_output(self, state, Q_encodings, is_training, fact_logits, context, targets=None):
        nbatch = tf.shape(state)[0]
        flength = int(fact_logits.shape[2])
        nqs = tf.shape(Q_encodings)[1]
        As = tf.zeros((nbatch, nqs, 0), dtype=tf.int32)
        fact_logits = tf.transpose(fact_logits, [0, 2, 1, 3]) #batch, words within facts, facts, vocab
        losses = tf.zeros((nbatch, nqs), dtype=tf.float32)
        for i in range(self.alength):
            if i > 0:
                A_encodings = self.encode_many(As, is_training, context=context[i]["encoding"])
                Q_and_A = tf.concat([Q_encodings, A_encodings], axis=2)
            else:
                Q_and_A = Q_encodings
            Q = tf_utils.fully_connected(Q_and_A, (flength, self.nh), context=context[i]["Q"])
            #batch, questions, words within facts, hidden
            K = tf_utils.fully_connected(state, (flength, self.nh), context=context[i]["K"])
            #batch, facts, words within facts, hidden
            Q = tf.transpose(Q, [0, 2, 1, 3]) #batch, words within facts, questions, hidden
            K = tf.transpose(K, [0, 2, 1, 3]) #batch, words within facts, facts, hidden
            copy_logits_by_index = tf_utils.attention(Q, K, fact_logits, mask=False, logits=True)
            #batch, words within facts, questions, vocab
            copy_logits_by_index = tf.transpose(copy_logits_by_index, [0, 2, 1, 3])
            #batch, questions, words within facts, vocab
            mix_logits = (context[i].sequential(Q_and_A, name="copy_index").
                    fully_connected((self.nh,)).relu().
                    fully_connected((flength + 1,))).value #batch, questions, words within facts + 1
            output_logits = (context[i].sequential(Q_and_A, name="output").
                    fully_connected((self.nh,)).relu().
                    fully_connected((1, self.nvocab,))).value #batch, questions, 1, vocab
            logits_to_mix = tf.concat([copy_logits_by_index, output_logits], axis=2)
            #batch, questions, words within facts + 1, vocab
            logits = tf_utils.mix_logits(mix_logits, logits_to_mix) #batch, questions, vocab
            if targets is None:
                next_w = multinomial(logits)
            else:
                next_w = targets[:,:,i]
                losses = losses + tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=next_w)
            As = tf.concat([As, tf.expand_dims(next_w, axis=2)], axis=2)
        if targets is None:
            return As
        else:
            return losses

    def answer(self, state, facts, Qs, is_training=tf.constant(False), context=None, targets=None):
        if context is None: context = self.context["answer"]
        flength = int(facts.shape[2])
        cells = context.sequential(self.encode_many(Qs, is_training=is_training, context=context["encode"]))
        for i in range(self.answer_depth):
            cells .transformer_cell(self.nheads, y=state, scale_weights=self.scale_weights, training=is_training, p_drop=self.p_drop)
        Q_encodings = cells.value
        onehot_facts = tf.one_hot(facts, depth=self.nvocab) #batch, facts, words within facts, vocab
        fact_logits = (onehot_facts - 1) * 1e9
        As = self.generate_output(state, Q_encodings, is_training, fact_logits, context=context["decode"])
        if targets is not None:
            losses = self.generate_output(state, Q_encodings, is_training, fact_logits, targets=targets, context=context["decode"])
            return As, losses
        return As
