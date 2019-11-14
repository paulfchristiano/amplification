import tensorflow as tf

def flatten_first(x):
    s = tf.shape(x)
    a, b, r = s[0], s[1], s[2:]
    flat_x = tf.reshape(x, tf.concat([(a*b,), r], axis=0))
    flat_x.set_shape(tf.TensorShape([x.shape[0] * x.shape[1]]).concatenate(x.shape[2:]))
    def unflatten(f):
        n = tf.shape(f)
        ab, r = n[0], n[1:]
        result = tf.reshape(f, tf.concat([(a, b), r], axis=0))
        result.set_shape(x.shape[:2].concatenate(f.shape[1:]))
        return result
    return flat_x, unflatten

def flatten_to(x, dims):
    unflatteners = []
    while len(x.shape) > dims:
        x, unflattener = flatten_first(x)
        unflatteners.append(unflattener)
    def unflatten(f):
        for unflattener in reversed(unflatteners):
            f = unflattener(f)
        return f
    return x, unflatten

def multinomial(logits):
    flat_logits, unflatten = flatten_to(logits, 2)
    flat_result = tf.squeeze(tf.multinomial(flat_logits, 1), axis=[-1])
    result = unflatten(flat_result)
    return tf.cast(result, tf.int32)
