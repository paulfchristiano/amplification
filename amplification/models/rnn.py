import tensorflow as tf

import amplification.tf_utils as tf_utils


class RNNModel(tf_utils.Model):
    def set_params(self, nvocab, nh=1000, ne=50, rnn_cell_type="gru", max_n_facts=None):
        self.nh = nh
        self.ne = ne
        self.nvocab = nvocab
        self.rnn_cell_type = rnn_cell_type
        self.max_n_facts = max_n_facts

    @property
    def rnn_cell(self):
        if self.rnn_cell_type == "gru":
            return tf_utils.gru_cell
        elif self.rnn_cell_type == "lstm":
            return tf_utils.lstm_cell_combined
        else:
            raise ValueError(self.rnn_cell_type)

    def embed(self, ws, name=None):
        context = self.context
        if name is not None: context = context[name]
        return tf_utils.embed(ws,
                              self.nvocab + 1,
                              self.ne,
                              context=context["embedding"])

    def initial_state(self, nbatch, name=None):
        context = self.context
        if name is not None: context = context[name]
        zeros = tf.zeros((self.nh,), tf.float32)
        initial_state = context.variable("init", zeros, tags=["trainable"])
        return (tf.ones((nbatch, self.nh), tf.float32) * initial_state)

    def run(self,
            num_steps=None,
            initial_state=None,
            inputs=None,
            input_ws=(),
            get_output=False,
            name=None,
            nbatch=None,
            temperature=1,
            do_dropout=False):
        initial_inputs = inputs
        context = self.context
        if name is not None: context = context[name]
        if input_ws != ():
            nbatch = tf.shape(input_ws)[0]
            num_steps = input_ws.get_shape()[1]
        if inputs is None:
            inputs = tf.zeros((nbatch, int(num_steps), 0))
        else:
            if nbatch is None:
                nbatch = tf.shape(inputs)[0]
            num_steps = inputs.get_shape()[1]

        if input_ws != () and not get_output:
            embedded_input_ws = self.embed(input_ws, name=name)
            inputs = tf.concat([inputs, embedded_input_ws], axis=2)
        if initial_state is not None and nbatch is None:
            nbatch = tf.shape(initial_state)[0]
        assert nbatch is not None
        assert num_steps is not None
        assert (get_output or (inputs is not None))
        if initial_state is None:
            initial_state = self.initial_state(nbatch, name=name)

        def step(inputs, prev):
            w, state, done = prev
            x, input_w = inputs
            if get_output:
                embedded_w = self.embed(w)
                x = tf.concat([x, embedded_w], axis=1)
            new_state = self.rnn_cell(x, state, context=context["rnn"])
            state = tf.where(done, state, new_state)
            #done = tf.logical_or(tf.equal(w, self.encoder[STOP]), done)
            if get_output:
                logit = tf_utils.fully_connected(state,
                                                 (self.nvocab, ),
                                                 context=context["output"])
                if input_w != ():
                    w = input_w
                else:
                    w = tf_utils.sample_logits(logit / temperature)
                w.set_shape(prev[0].get_shape())
                    #w = tf.where(done, self.encoder[PADDING] * tf.ones_like(w), w)
                return (w, logit), (w, state, done)
            else:
                return (), (w, state, done)

        not_done = tf.zeros((nbatch, ), tf.bool)
        start_word = self.nvocab * tf.ones((nbatch, ), dtype=tf.int32)
        output_dtypes = (tf.int32, tf.float32) if get_output else ()
        result = tf_utils.for_loop(step, num_steps,
                                   (start_word, initial_state, not_done),
                                   (inputs, input_ws), output_dtypes)
        if get_output:
            (ws, logits), (_, final_state, _) = result
            return ws, logits, final_state
        else:
            _, (_, final_state, _) = result
            return final_state
