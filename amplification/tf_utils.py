from inspect import signature
import random
import string
import pickle
from copy import copy
import re
import os

import tensorflow as tf
from tensorflow.python.ops import array_ops, tensor_array_ops, control_flow_ops
import numpy as np
import scipy
from decorator import decorator
from collections import defaultdict

#===== Context ========

class Context(object):
    """
    A dict of dicts that can be used for variable sharing

    d: the contents
    tags: for each non-Context member, we store a list of tags such as 'trainable' or 'regularizer'
    (all variables are automatically tagged "variable")
    frozen: if True, this Context can't be modified
    (attempts to modify the Context will appear to succeed iff they are replaying previous operations)
    path: tries to store the names of all parents of this context, for debugging purposes
    _is_a_context: used in place of isinstance(., Context) to work better interactively
    """
    def __init__(self, load_from=None, path="root"):
        self.frozen = False
        self.d =  {}
        self.tags = defaultdict(dict)
        self.path = path #used only for debugging, it's fine if two things have the same path
        self._is_a_context = True
        if load_from is not None:
            self.load(load_from)

    def freeze(self):
        """cause further modifications to do nothing"""
        if not self.frozen:
            self.frozen = True
            for v in self.d.values():
                if is_context(v): v.freeze()

    def load(self, d):
        """recursively populate this object with the elements from a dictionary or Context d"""
        result = {}
        for k, v in d.items():
            if 'variable' in self.tags[k]:
                result[self.tags[k]["variable"]] = v
            elif isinstance(v, dict) or is_context(v):
                result.update(self[k].load(v))
            else:
                self[k] = v
        if is_context(d):
            for k, tags in d.tags.items():
                self.tags[k].update(tags)
        return result

    def variables(self):
        return self.map(filter=lambda x,t: "variable" in t)

    def __setitem__(self, k, v):
        self.set(k, v)

    def set(self, k, v, tags={}):
        tags = to_tags(tags)
        if isinstance(v, tf.Variable):
            tags["variable"] = tf.placeholder(dtype=v.dtype, shape=v.get_shape())
        if self.frozen:
            assert k in self.d
        else:
            if is_context(v):
                self.d[k].load(v)
            else:
                self.d[k] = v
                self.tags[k].update(tags)

    def __getitem__(self, k):
        if self.frozen:
            assert k in self.d
        if k not in self.d:
            self.d[k] = Context(path="{}/{}".format(self.path, k))
        return self.d[k]

    def items(self):
        return self.d.items()

    def keys(self):
        return self.d.keys()

    def __contains__(self, key):
        return key in self.d

    def _iter__(self):
        return self.d.__iter__()

    def collect(self, map=lambda x, tags : x, filter=lambda x, tags:True):
        """return a list of all values in this Context and its descendants
        """
        result = []
        def add_to_result(x, tags):
            if filter(x, tags):
                result.append(map(x, tags))
        self.map(add_to_result)
        return result

    def zip_with(self, other, zip=lambda xs, tags : xs, filter=lambda xs, tags:True, tag_zip=lambda xs, tags : tags[0]):
        if tag_zip is None:
            def tag_zip(xs, tags):
                result = {}
                for t in tags:
                    for k in t:
                        result[k] = t[k]
        result = Context()
        for k, v in self.d.items():
            if is_context(v):
                result.d[k] = v.zip_with(other.d[k], zip, filter, tag_zip)
                result.tags[k] = copy(self.tags[k])
            else:
                xs = (self.d[k], other.d[k])
                tags = (self.tags[k], other.tags[k])
                if filter(xs, tags):
                    result.d[k] = zip(xs, tags)
                    result.tags[k] = tag_zip(xs, tags)
        return result

    def map(self, map=lambda x, tags : x, filter=lambda x, tags:True, tag_map=lambda x, tags: copy(tags)):
        result = Context()
        for k, v in self.d.items():
            if is_context(v):
                result.d[k] = v.map(map, filter)
                result.tags[k] = copy(self.tags[k])
            elif filter(v, self.tags[k]):
                result.d[k] = map(v, self.tags[k])
                result.tags[k] = tag_map(v, self.tags[k])
        return result

    def variable(self, name, value, tags={}):
        """create a variable with a given name, value, and tags

        if this is frozen, instead retrieve an existing variable with the given name
        """
        if name in self.d:
            return self.d[name]
        else:
            assert not self.frozen
            var = get_variable("{}/{}".format(self.path, name), value)
            tags = to_tags(tags)
            self.set(name, var, tags=tags)
            return var

    def sequential(self, initial_value, name="sequential"):
        context = self[name]
        return Sequential(initial_value, context)

    def counter(self, name, cap=None):
        """creates a new variable that increments by 1 each time it accessed"""
        v = self.variable(name, 0)
        with tf.control_dependencies([v.assign_add(1)]):
            return tf.identity(v) if cap is None else tf.minimum(int(cap), v)

    def dict(self):
        result = {}
        for k, v  in self.items():
            result[k] = v.dict() if is_context(v) else v
        return result

    def __repr__(self):
        return "Context({})".format(", ".join("{}: {}".format(k, v) for k, v in self.items()))

    def __getstate__(self):
        result = copy(self.__dict__)
        result['d'] = dict(result['d'])
        result['tags'] = dict(result['tags'])
        return result

    def __setstate__(self, state):
        self.__dict__ = state
        self.d = defaultdict(Context, self.d)
        self.tags = defaultdict(dict, self.tags)

def to_tags(tags):
    result = {}
    if isinstance(tags, dict):
        result.update(tags)
    else:
        for tag in tags:
            result[tag] = True
    return result

def get_variable(name, value):
    def dtype(x):
        if isinstance(x, tf.Tensor):
            return x.dtype
        else:
            return np.asarray(x).dtype
    def shrink(t):
        if t == np.int64:
            return np.int32
        elif t == np.float64:
            return np.float32
        else:
            return t
    return tf.Variable(value, name=name, dtype=shrink(dtype(value)))

def is_context(v):
    return hasattr(v, "_is_a_context")

def needs_context(f):
    return 'context' in signature(f).parameters.keys()

def call_with_context(f, context, *args, **kwargs):
    """call f with the given arguments, including context as a kwarg if necessary"""
    if needs_context(f):
        assert "context" not in kwargs
        kwargs["context"] = context
    return f(*args, **kwargs)

@decorator
def contextual(f, *args,  **kwargs):
    """freeze the context after calling f"""
    context = signature(f).bind(*args, **kwargs).arguments["context"]
    assert context is not None
    result = f(*args, **kwargs)
    context.freeze()
    return result

@decorator
def maybe_contextual(f, *args, **kwargs):
    """freeze the context after calling f, but the context is not mandatory"""
    context = signature(f).bind(*args, **kwargs).arguments["context"]
    if context is None:
        context = Context()
        kwargs["context"] = context
    result = f(*args, **kwargs)
    context.freeze()
    return result

#=======Convenient calling=========

class Caller(object):
    def __init__(self, v):
        self.value = v

    def __getattr__(self, attr):
        if attr in registered_names:
            f = registered_names[attr]
            return lambda *args, **kwargs: self.call(f, *args, **kwargs)
        elif attr in macro_names:
            f = macro_names[attr]
            return lambda * args, **kwargs: f(self, *args, **kwargs)
        else:
            raise AttributeError(attr)

    def call(self, f, *args, **kwargs):
        return Caller(f(self.value, *args, **kwargs))

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.value)

class Sequential(Caller):
    """object for conveniently calling a sequence of operations

    calls each successive operation in a new sub-context indexed by 1, 2, ...
    """
    def __init__(self, initial_value, context):
        self.value = initial_value
        self.context = context
        self.current_layer = 0

    def call(self, f, *args, **kwargs):
        context = self.context[self.current_layer]
        self.value = call_with_context(f, context, self.value, *args, **kwargs)
        self.current_layer += 1
        return self

registered_names = {}
def register(f):
    """add f as a method to all Caller objects

    f will be given the Caller's value as its first argument
    """
    registered_names[f.__name__] = f
    return f

macro_names = {}
def macro(f):
    """add f as a method to all Caller objects

    f will be given the Caller itself as its first argument
    this is useful for e.g. chaining together several registered functions
    """
    macro_names[f.__name__] = f
    return f


#=====Common operations=========

shape = register(tf.shape)
relu = register(tf.nn.relu)
tanh = register(tf.tanh)
sigmoid = register(tf.nn.sigmoid)
concat = register(tf.concat)
softmax = register(tf.nn.softmax)
softplus = register(tf.nn.softplus)

def log_partition(x, keep_dims=False, axis=-1):
    m = tf.reduce_max(x, axis=axis, keep_dims=True)
    log_sum_exp = tf.log(tf.reduce_sum(tf.exp(x-m), axis=axis, keep_dims=True))
    result = log_sum_exp + m
    if not keep_dims:
        result = tf.squeeze(result, [axis])
    return result

#logits -> log(p)
def normalize_logits(logits):
    return logits - log_partition(logits, keep_dims=True)

# w : ... x components
# logits: ... x components x elements
def mix_logits(w, logits):
    log_ps = normalize_logits(logits)
    expanded_w = tf.expand_dims(w, -1)
    return log_partition(expanded_w + log_ps, axis=-2) - log_partition(expanded_w, axis=-2)

def softsum(x, y):
    return y + softplus(x - y)

#compute log [ sigmoid(s) * softmax(x) + (1 - sigmoid(s)) * softmax(y) ]
# s:  ...
# x: ... x elements
# y: ... x elements
def soft_combine(x, y, s):
    w = tf.stack([s, tf.zeros_like(s)], axis=-1)
    logits = tf.stack([x, y], axis=-2)
    return mix_logits(w, logits)

@register
def reshape(x, dims):
    if isinstance(dims, tf.Tensor):
        return tf.reshape(x, dims)
    def prep_dim(dim):
        if isinstance(dim, tf.Dimension):
            val = dim.value
            return val if val is not None else -1
        return dim
    return(tf.reshape(x, [prep_dim(dim) for dim in dims]))

#NOTE: some of these implementations closely follow the prettytensor versions
#https://github.com/google/prettytensor

@register
def leaky_relu(x, alpha=0.01):
    return tf.where(tf.less(x, 0.0), alpha*x, x)

def tensordot(x, y, axes):
    x_shape = tf.shape(x)[:-axes] if axes > 0 else tf.shape(x)
    y_shape = tf.shape(y)[axes:]
    rx = tf.reshape(x, tf.stack([tf.reduce_prod(x_shape), -1]))
    ry = tf.reshape(y, tf.stack([-1, tf.reduce_prod(y_shape)]))
    rresult = tf.matmul(rx, ry)
    return tf.reshape(rresult, tf.concat([x_shape, y_shape], axis=0))

@register
@contextual
def fully_connected(x, outshape, l2loss=0.0, context=None, init_relative_reg=True, scale_weights=True, weight_factor=1, in_axes=1):
    inshape = tuple([int(d) for d in x.shape[-in_axes:]])
    scale = 1 / np.sqrt(np.prod(inshape) + 0.01)
    if scale_weights: weight_factor *= scale
    init = weight_factor * normal_init(inshape + outshape) if not context.frozen else None
    W = context.variable("W", init, tags=['trainable'])
    b = context.variable("b", np.zeros(outshape), tags=['trainable', 'bias'])
    xW = scale / weight_factor * tensordot(x, W, in_axes)
    random_W = weight_factor * tf.random_normal(inshape + outshape)
    context.set("drift", lambda eps : W.assign_add(eps * (random_W - W)), tags=["drift", "nosave"])
    if l2loss > 0 and not context.frozen:
        init_W = context.variable("init_W", init, tags=['init'])
        weights_to_penalize = (W-init_W) if init_relative_reg else W
        context.set("l2_penalty", l2loss * tf.reduce_mean(weights_to_penalize**2), tags=['regularizer', 'nosave'])
    assert xW.shape.dims is not None
    return xW+b

def ortho_init(shape, insize=None, outsize=None, get_scale=False):
    if insize is None:
        insize = shape[0]
    if outsize is None:
        outsize = shape[1]
        assert len(shape) == 2
    #lasagne ortho init for tf
    if len(shape) == 2:
        flat_shape = shape
    elif len(shape) == 4: # assumes NHWC
        flat_shape = (np.prod(shape[:-1]), shape[-1])
    else:
        raise NotImplementedError
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    result = q[:shape[0], :shape[1]].astype(np.float32)
    scale = 1 / np.sqrt(np.maximum(insize, outsize))
    return (result, scale) if get_scale else result

@register
@contextual
def conv2d(x, out_channels, kernel_shape, stride, context=None, init_relative_reg=True, l2loss=0.0):
    if isinstance(kernel_shape, int): kernel_shape = (kernel_shape, kernel_shape)
    assert x.shape.ndims == 4
    in_channels = int(x.shape.dims[3])
    assert in_channels is not None
    kernel_size = np.prod(kernel_shape)
    filter_shape = kernel_shape + (in_channels, out_channels)
    scale = 1 / np.sqrt(kernel_size * in_channels)
    init = normal_init(filter_shape)
    W = context.variable("W",
        init,
        tags=["trainable"]
    )
    random_W = tf.random._normal(filter_shape)
    full_stride = [1, stride, stride, 1] if isinstance(stride, int) else [1, stride[0], stride[1], 1]
    xW = tf.nn.conv2d(x, W, full_stride, padding="SAME") * scale
    b = context.variable("b", tf.zeros(xW.shape[1:]), tags=['trainable', 'bias'])
    context.set("drift", lambda eps : W.assign_add(eps * (random_W - W)), tags=["drift", "nosave"])
    if l2loss > 0 and not context.frozen:
        init_W = context.variable("init_W", init, tags=['init'])
        weights_to_penalize = (W-init_W) if init_relative_reg else W
        context.set("l2_penalty", l2loss * tf.reduce_mean(weights_to_penalize**2), tags=['regularizer', 'nosave'])
    return xW+b

def normal_init(shape):
    return np.random.randn(*shape)

def masked_shape(x, masked_axes=(), as_tensor=False):
    dims = []
    x_shape = shape(x) if as_tensor else x.get_shape()
    for i in range(x.shape.ndims):
        if i in masked_axes:
            dims.append(tf.constant(1) if as_tensor else 1)
        else:
            dims.append(x_shape[i] if as_tensor else x_shape[i].value)
    return tf.stack(dims) if as_tensor else dims

#XXX factor this into another file
def automap(*to_map):
    def decorate(f):
        def wrapped(*args, **kwargs):
            kwargs = signature(f).bind(*args, **kwargs).arguments
            prototype = kwargs[to_map[0]]
            mapped = isinstance(prototype, tuple) or isinstance(prototype, dict)
            if isinstance(prototype, tuple):
                arity = len(prototype)
                for k in to_map:
                    assert isinstance(kwargs[k], tuple) and len(kwargs[k]) == arity
                def map_over(i):
                    new_kwargs = copy(kwargs)
                    for k in to_map: new_kwargs[k] = kwargs[k][i]
                    return wrapped(**new_kwargs)
                return tuple(map_over(i) for i in range(arity))
            #if isinstance(prototype, dict):
            #    keys = prototype.keys()
            #    for k in to_map:
            #        assert isinstance(kwargs[k], dict) and kwargs[k].keys() == keys
            #    def map_over(key):
            #        new_kwargs = copy(kwargs)
            #        for k in to_map: new_kwars[k] = kwargs[k][key]
            #        return wrapped(**new_kwargs)
            #    return {key:map_over(key) for key in keys}
            else:
                return f(**kwargs)
        return wrapped
    return decorate

def swap(x):
    perm = list(range(len(x.shape)))
    perm[0] = 1
    perm[1] = 0
    return tf.transpose(x, perm)

def set_dim(x, axis, size):
    new_shape = [None for _ in x.shape]
    new_shape[axis] = size
    x.set_shape(new_shape)

@automap("ta")
def read_arrays(ta, t):
    return ta.read(t)

@automap("ta", "x")
def write_arrays(ta, t, x):
    return ta.write(t, x)

@automap("ta", "x")
def fill_arrays(ta, x):
    return ta.unstack(swap(x))

@automap("x")
def make_arrays(x, T):
    return fill_arrays(arrays_to_hold(x.dtype, T), x)

@automap("ta")
def dump_arrays(ta, T):
    result = swap(ta.stack())
    try:
        set_dim(result, axis=1, size=T)
    except TypeError:
        pass
    return result

@automap("dtype")
def arrays_to_hold(dtype, T):
    return tensor_array_ops.TensorArray(dtype=dtype, size=T)

@automap("x")
def index_tensors(x, t):
    return x[:, t]

@automap("xs")
def stack_lists(xs):
    return tf.stack(xs, axis=1)

@automap("dtype")
def lists_to_hold(dtype):
    return []

@automap("x", "l")
def append_to_lists(x, l):
    l.append(x)

def for_loop(body, timesteps, initial_state, inputs, output_dtypes, unroll=False, **kwargs):
    if unroll:
        assert isinstance(timesteps, int), "If you want to use an unrolled for loop, your timesteps must be an int"
        state = initial_state
        outputs = lists_to_hold(output_dtypes)
        for time in range(timesteps):
            output, state = body(index_tensors(inputs, time), state)
            append_to_lists(output, outputs)
        return stack_lists(outputs), state
    else:
        input_arrays = make_arrays(inputs, timesteps)
        # output_arrays = tensor_array_ops.TensorArray(dtype=output_dtypes, size=timesteps)
        output_arrays = arrays_to_hold(output_dtypes, timesteps)
        time = array_ops.constant(0, dtype=tf.int32, name="time")
        def loop_body(time, output_arrays, state):
            input = read_arrays(input_arrays, time)
            output, new_state = body(input, state)
            output_arrays = write_arrays(output_arrays, time, output)
            return (time+1, output_arrays, new_state)
        body(index_tensors(inputs, 0), initial_state)
        _, output_arrays, final_state = control_flow_ops.while_loop(
            cond=lambda time, output_arrays, state: time < timesteps,
            body=loop_body,
            loop_vars=(time, output_arrays, initial_state),
            **kwargs
        )
    return dump_arrays(output_arrays, timesteps), final_state

@register
@contextual
def embed(x, vocab, outsize, context=None):
    embedding_init = normal_init((vocab, outsize)) if not context.frozen else None
    embedding = context.variable("embedding_vectors", embedding_init, tags=["trainable"])
    return tf.nn.embedding_lookup(embedding, x)

try:
    from blocksparse.norms import layer_norm as _layer_norm
    def layer_norm(x, gain, bias, axis=1, epsilon=1e-5, relu=False):
        return _layer_norm(x, gain, bias, axis=axis, epsilon=epsilon, relu=relu)
except ImportError: #fall back to a slow layer norm
    print("unable to load custom GPU layer_norm, using slow implementation...")
    def layer_norm(x, gain, bias, axis=1, epsilon=1e-5, relu=False):
        m, v = tf.nn.moments(x, axes=(axis,), keep_dims=True)
        x = (x - m) / tf.sqrt(v + epsilon)
        x = x * gain + bias
        if relu: x = tf.nn.relu(x)
        return x

@register
@contextual
def trainable_layer_norm(x, size, axis=1, context=None):
    gain = context.variable("gain", np.ones((size,), dtype=np.float32), tags=["trainable"])
    bias = context.variable("bias", np.zeros((size,), dtype=np.float32), tags=["trainable"])
    return layer_norm(x, gain, bias, axis=axis, relu=False)

def sample_logits(logits, axis=1):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.cast(tf.argmax(logits - tf.log(-tf.log(noise)), axis), tf.int32)

def mask_past(w, mask_offset=0):
    #kill all of the terms with second coordinate > first coordinate
    hide = (1 - tf.matrix_band_part(tf.ones_like(w), -1, tf.cast(mask_offset, tf.int64)))
    return w - 1e9 * hide

#second to last dim indexes different cells
#last dim indexes vectors
#(last dim of v are logits if logits=True)
@register
def attention(q, k, v, mask=False, mask_offset=0, logits=False):
    w = tf.matmul(q, k, transpose_b=True)
    nh = tf.cast(q.get_shape()[-1], tf.float32)
    w = w / tf.sqrt(nh)
    if mask: w = mask_past(w, mask_offset)
    if logits:
        # in this case v is logits :  ... x cells x elements
        # w : ... x cells x cells
        # want to use the same v for every cell : 1 x cells x elements
        return mix_logits(w, tf.expand_dims(v, axis=-3)) 
    else:
        w = tf.nn.softmax(w)
        return tf.matmul(w, v)

#in: batch x cells x size
#out: batch x heads x cells x size
def split_heads(x, nheads, headsize, context=None, scale_weights=True):
    split_x = fully_connected(x,
            outshape = (nheads, headsize),
            context=context["split"], scale_weights=scale_weights)
    split_x = tf.transpose(split_x, [0, 2, 1, 3])
    return split_x

#in: batch x heads x cells x headsize
#out: batch x cells x size
def merge_heads(split_x, size, context=None, scale_weights=True):
    split_x = tf.transpose(split_x, [0, 2, 1, 3])
    x = fully_connected(split_x, outshape=(size,), in_axes=2, context=context["merge"],
            scale_weights=scale_weights)
    return x

@register
@contextual
def multi_attention(q, k, v, nheads, context=None, attention_args={}, **kwargs):
    size = int(q.get_shape()[2])
    assert size % nheads == 0
    head_size = size // nheads
    split_q = split_heads(q, nheads, head_size, context=context["Q"], **kwargs)
    split_k = split_heads(k, nheads, head_size, context=context["K"], **kwargs)
    split_v = split_heads(v, nheads, head_size, context=context["V"], **kwargs)
    split_result = attention(split_q, split_k, split_v, **attention_args)
    return merge_heads(split_result, size, context=context["result"], **kwargs)

#in/out: batch x cells x size
@register
@contextual
def transformer_cell(x, nheads, y=None, context=None, training=None, scale_weights=True, p_drop=0.1, **kwargs):
    size = int(x.get_shape()[2])
    if y is None: y = x
    attended = multi_attention(x, y, y, nheads, context=context["attention"], scale_weights=scale_weights, **kwargs)
    attended = dropout(attended, p_drop=p_drop, do_dropout=training, shared_axes=(1,))
    x = trainable_layer_norm(x + attended, size=size, axis=2, context=context["ln1"])
    mlp = (context.sequential(x).
               fully_connected(outshape=(size*4,), scale_weights=scale_weights).relu().
               fully_connected(outshape=(size,), scale_weights=scale_weights).
               batch_norm(axes=(0, 1), training=training)).value
    mlp = dropout(mlp, p_drop=p_drop, do_dropout=training, shared_axes=(1,))
    x = trainable_layer_norm(x + mlp, size=size, axis=2, context=context["ln2"])
    return x

#following alec radford's implementation
@register
@contextual
def lstm_cell(x, c, h, context=None, **kwargs):
    nh = h.shape[1].value
    z = sum(
        s.fully_connected((4*nh,), **kwargs).trainable_layer_norm(4*nh).value
        for s in [context.sequential(h, "h"), context.sequential(x, "x")]
    )
    i, f, o, u = tf.split(z, 4, axis=1)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f)
    o = tf.nn.sigmoid(o)
    u = tf.tanh(u)
    c = f*c + i*u
    h = o*tf.tanh(trainable_layer_norm(c, nh, context=context["c"]))
    return c, h

@register
@contextual
def lstm_cell_combined(x, s, context=None, **kwargs):
    c, h = tf.split(s, 2, axis=1)
    c, h = lstm_cell(x, c, h, context=context, **kwargs)
    return tf.concat([c, h], axis=1)

@register
@contextual
def gru_cell(x, h, context=None, do_dropout=False, dropout_p=0.5, **kwargs):
    nh = h.shape[1].value
    y = sum(
        s.fully_connected((2*nh,), **kwargs).trainable_layer_norm(2*nh).value #XXX should I have this layer norm? I think not
        for s in [context.sequential(h, "h"), context.sequential(x, "x")]
    )
    r, z = tf.split(y, 2, axis=1)
    r = tf.nn.sigmoid(r)
    z = tf.nn.sigmoid(z - 1)
    update = sum(
        s.fully_connected((nh,), **kwargs).trainable_layer_norm(nh).value
        for s in [context.sequential(r*h, "rh"), context.sequential(x, "x2")]
    )
    update_h = tf.tanh(update)
    update_h = dropout(
            update_h, p_drop=dropout_p,
            do_dropout=do_dropout)
    h = h + z * (update_h - h)
    h = trainable_layer_norm(h, nh, context=context["ln"])
    return h

@register
def dropout(x, p_drop=0.5, do_dropout=True, shared_axes=()):
    p = 1 - p_drop
    #replace shared_axes with 1
    #so we broadcast in that dimension, sharing dropout mask
    mask_shape = masked_shape(x, shared_axes, as_tensor=True)
    mask = tf.floor(tf.random_uniform(mask_shape) + p)
    if isinstance(do_dropout, bool):
        return mask*x/p if do_dropout else x
    else:
        return tf.cond(do_dropout, lambda : mask*x / p, lambda : x)

@register
def cnn_dropout(*args, **kwargs):
    return dropout(*args, shared_axes=(1, 2), **kwargs)

@register
def add_gaussian_noise(x, std, add_noise=tf.constant(True)):
    def noisy():
        noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=std, dtype=tf.float32)
        return x + noise
    return tf.cond(add_noise, noisy, lambda : x)

@register
@contextual
def batch_norm(x, training=True, scale=False, horizon=300, variance_epsilon=0.001, axes=(0,), context=None):
    if type(training) is bool:
        training = tf.constant(training)
    stat_dims = masked_shape(x, axes, as_tensor=False)
    m, v = tf.nn.moments(x, axes=axes, keep_dims=True)
    running_m = context.variable("running_m", np.zeros(stat_dims), tags=["stats"])
    running_v = context.variable("running_v", np.ones(stat_dims), tags=["stats"])
    #TODO might want to separate this out into a different operation
    learned_m = context.variable("learned_m", np.zeros(stat_dims), tags=["trainable"])
    learned_v = context.variable("learned_v", np.ones(stat_dims), tags=["trainable"]) if scale else 1.0
    #TODO this is a hacky way to make sure this gets run whenever we run through this layer
    #if this has a negative impact on performance we should figure out something better
    def normalize(x, old_m, old_v): return (x - old_m) * tf.sqrt(learned_v / (old_v + variance_epsilon)) + learned_m
    def training_val():
        update_m_op = running_m.assign_add((m - running_m) / horizon)
        update_v_op = running_v.assign_add((v - running_v) / horizon)
        with tf.control_dependencies([update_m_op, update_v_op]):
            return normalize(x, m, v)
    def test_val():
        return normalize(x, running_m, running_v)
    return tf.cond(training, training_val, test_val)

@register
@contextual
def cnn_batch_norm(*args, context=None, **kwargs):
    return batch_norm(*args, axes=(0,1,2), context=context, **kwargs)

def print_variance(x, axes=(0,), name=None, flatten=False):
    if isinstance(axes, int):
        axes = (axis,)
    to_print = []
    if name is not None: to_print.append(name)
    if flatten:
        z = tf.reshape(x, [-1])
    else:
        z = x
    for axis in axes:
        m, v = tf.nn.moments(z, axes=(axis,), keep_dims=False)
        mean_v = tf.reduce_mean(v)
        to_print.append(mean_v)
    return tf.Print(x, to_print)

def drift_ops(context, epsilon):
    return [f(epsilon) for f in context.collect(filter = lambda x, tags : "drift" in tags)]

@contextual
def minimize(loss, variables, optimizer_factory=tf.train.AdamOptimizer, context=None,
        clip_grads=None, filter=None, optimizer_args={}, print_everything=False,
        **kwargs):
    if filter is None:
        filter = lambda x, tags: "trainable" in tags
    var_list = variables.collect(filter=filter)
    with store_variables_in(context):
        if not context.frozen:
            context["optimizer"] = optimizer_factory(**optimizer_args)
        grads = tf.gradients(loss, var_list)
        if clip_grads is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, clip_grads)
        grads = list(zip(grads, var_list))
        old_vars = [v.read_value() + 0 for v in var_list]
        if print_everything:
            with tf.control_dependencies(old_vars):
                minimize_op = context["optimizer"].apply_gradients(grads, **kwargs)
            with tf.control_dependencies([minimize_op]):
                print_diffs = [print_variance(v.read_value() - old_v, flatten=True, name="{} diff".format(v.name))
                               for v, old_v in zip(var_list, old_vars)]
                print_originals = [print_variance(g, name="{} grad".format(v.name), flatten=True) for v, g in grads]
                print_weights = [print_variance(v, flatten=True, name="{}".format(v.name)) for v in var_list]
            with tf.control_dependencies(print_diffs + print_originals + print_weights):
                return tf.no_op()
        else:
            return context["optimizer"].apply_gradients(grads, **kwargs)

def moving_averages(variables, horizon, store_averages_in):
    """computes moving averages

    variables -- a Context storing the variables to be averaged
    horizon -- the horizon to average over
    store_averages_in -- a Context where the resulting variables will be stored,
    with a structure matching variables
    """
    updates = []
    for k, v in variables.items():
        if is_context(v):
            updates.extend(moving_averages(v, horizon, store_averages_in=store_averages_in[k]))
        else:
            ema_v = store_averages_in.variable(k, v.initialized_value())
            updates.append(ema_v.assign_add((v - ema_v) / tf.cast(horizon, tf.float32)))
    return updates

#======Saving========

class Model(object):
    def __init__(self, initial_values=None, context=None, **params):
        self.context = context or Context(path=self.__class__.__name__)
        self.params = params
        self.set_params(**params)
        self.initial_values = initial_values
        self.initialized = False

    def set_params(self, **kwargs):
        raise NotImplementedError()

    def initialize(self, session=None):
        assert not self.initialized
        self.context.freeze()
        self.sess = session or tf.get_default_session()
        assert self.sess is not None
        self.initializer = tf.variables_initializer(self.variables)
        self.set_variables = tf.group(*self.context.collect(
                filter=lambda x, tags: "variable" in tags,
                map=lambda x, tags: x.assign(tags["variable"])
            ))
        if self.initial_values is None:
            self.sess.run(self.initializer)
            self.initialized = True
            self.initial_values = self.get_values()
        else:
            self.set_values(self.initial_values)
            self.initialized = True

    @property
    def variables(self):
        return self.context.collect(filter=lambda x, tags : "variable" in tags)

    def reset(self):
        self.set_values(self.initial_values)

    def set_values(self, other):
        feeds = self.context.load(other)
        self.sess.run(self.set_variables, feeds)

    def get_values(self):
        assert self.initialized
        values = self.sess.run(self.variables)
        m = {v:val for v, val in zip(self.variables, values)}
        return self.context.map(
                map = lambda x, tags: m[x] if "variable" in tags else x,
                filter = lambda x, tags: "nosave" not in tags
            )

    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self.__getstate__(), f)

    @classmethod
    def load(cls, path):
        result = cls.__new__(cls)
        with open(path, "rb") as f: state = pickle.load(f)
        result.__setstate__(state)
        return result

    def __getstate__(self):
        return self.params, self.get_values().dict()

    def __setstate__(self, state):
        params, values = state
        self.__init__(initial_values=values, **params)

#=======Misc.===========

class store_variables_in(object):
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        self.initial_all_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self.initial_trainable_vars = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def __exit__(self, *args):
        new_all_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        new_trainable_vars = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        all_vars = new_all_vars - self.initial_all_vars
        trainable_vars = new_trainable_vars - self.initial_trainable_vars
        non_trainable_vars = all_vars - trainable_vars
        for i, v in enumerate(sorted(trainable_vars, key=var_key)):
            self.context["trainable"].set(i, v, tags=["trainable"])
        for i, v in enumerate(sorted(non_trainable_vars, key=var_key)):
            self.context["non_trainable"].set(i, v)

def var_key(var):
    """
    produces keys used to sort variables

    if you run the same sequence of tensorflow instructions multiple times,
    this will sort the resulting variables in the same way
    """
    return tuple(to_int_if_possible(piece) for piece in split_alpha_and_num(var.name))

def to_int_if_possible(x):
    try:
        return int(x)
    except ValueError:
        return x

def split_alpha_and_num(s):
    return re.split(r'(\d+)', s)

def run_all(session, ops, **kwargs):
    def flatten(xs, start_index=0):
        if isinstance(xs, list) or isinstance(xs, tuple):
            result = []
            mapping = []
            next_index = start_index
            for x in xs:
                flat_x, x_mapping, next_index = flatten(x, next_index)
                result.extend(flat_x)
                mapping.append(x_mapping)
            mapping = type(xs)(mapping)
            return result, mapping, next_index
        elif isinstance(xs, dict):
            result = []
            mapping = {}
            next_index = start_index
            for k, v in xs.items():
                flat_x, x_mapping, next_index = flatten(v, next_index)
                mapping[k] = x_mapping
                result.extend(flat_x)
            return result, mapping, next_index
        else:
            result = [xs]
            mapping = start_index
            next_index = start_index + 1
            return result, mapping, next_index
    def unflatten(xs, mapping):
        if isinstance(mapping, list) or isinstance(mapping, tuple):
            return type(mapping)([unflatten(xs, m) for m in mapping])
        elif isinstance(mapping, dict):
            return {k: unflatten(xs, v) for k, v in mapping.items()}
        elif isinstance(mapping, int):
            return xs[mapping]
        else:
            raise ValueError
    op_list, mapping, _ = flatten(ops)
    result_list = session.run(op_list, **kwargs)
    result = unflatten(result_list, mapping)
    return result
