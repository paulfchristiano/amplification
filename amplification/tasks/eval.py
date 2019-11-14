import random
from collections import defaultdict

import numpy as np

from amplification.tasks.core import idk, is_known, uniform, Task, sequences

class BaseEvalTask(Task):
    fact_length = 3
    question_length = 2

    simple_query_a = 1
    simple_query_b = 2
    simple_question_tokens = [simple_query_a, simple_query_b]
    compound_query = 3
    fixed_vocab = 4

    def repr_symbol(self, x):
        f_str = self.repr_f_symbol(x)
        if f_str is not None: return f_str
        if x == 0: return '?'
        if x == self.simple_query_a: return 'A'
        if x == self.simple_query_b: return 'B'
        if x == self.compound_query: return 'C'
        if x in self.chars:
            return 'abcdefghijklmnopqrstuv'[x]
        raise ValueError(x)

    @property
    def interaction_length(self):
        return 4 + self.f_interaction_length
    
    @property
    def question_length(self):
        return max(1 + self.length, self.f_question_length)

    @property
    def answer_length(self):
        return max(self.length, self.f_answer_length)

    @property
    def fact_length(self):
        return max(3 * self.length, self.f_fact_length)

    def __init__(self, nchars=8, length=2, **f_args):
        self.size = nchars ** length
        self.nchars = nchars
        self.length = length
        self.nvocab = self.fixed_vocab
        self.chars = self.allocate(self.nchars)
        self.min_char = self.chars[0]
        self.max_char = self.chars[-1]
        self.vars = list(sequences(self.chars, length))
        self.make_values(**f_args)

    def are_chars(self, x):
        return np.logical_and(np.all(x >= self.min_char, axis=-1), np.all(x <= self.max_char, axis=-1))

    def make_dbs(self, difficulty=float('inf')):
        instance_size = min(difficulty+8, self.size)
        unused_vars = set(self.vars)
        values = {}
        depths = {}
        assignments = {}
        facts = []
        f_db, f_facts = self.make_f_db()
        facts.extend(f_facts)
        def choice(S): return random.choice(tuple(S))
        for _ in range(instance_size):
            x = choice(unused_vars)
            if len(values) < int(np.sqrt(instance_size)):
                val = self.sample_val(f_db)
                assignments[x] = [val]
                depths[x] = 0
                facts.append(x + (val,))
            else:
                y = choice(values)
                z = choice(values)
                assignments[x] = [y, z]
                depths[x] = max(depths[y], depths[z]) + 1
                facts.append(x + y + z)
                val, f_facts = self.apply_f(values[y], values[z], f_db)
                facts.extend(f_facts)
            values[x] = val
            unused_vars.remove(x)
        fast_db = {}
        fast_db["values"] = values
        fast_db["assignments"] = assignments
        fast_db["depths"] = depths
        fast_db["used_vars"] = np.array(list(values))
        fast_db["f"] = f_db
        return np.asarray([self.pad(fact, self.fact_length) for fact in facts]), fast_db

    def make_qs(self, nqs, fast_db):
        indices = np.random.randint(len(fast_db["used_vars"]), size=nqs)
        vs = fast_db["used_vars"][indices]
        queries = self.compound_query * np.ones((nqs, 1))
        padding = idk * np.ones((nqs, self.question_length - 1 - self.length))
        return np.concatenate([queries, vs, padding], axis=1).astype(np.int32)
    
    def recursive_answer(self, Q):
        Q = tuple(Q)
        if Q[0] == self.compound_query:
            vals = []
            for q in [self.simple_query_a, self.simple_query_b]:
                x = (yield None, (q,)  + Q[1:])
                if not self.are_chars(x[:self.length]):
                    yield self.pad(x), None
                    return
                vals.append((yield None, self.pad((self.compound_query,) + tuple(x[:self.length]), self.question_length)))
            result = (yield from self.recursive_apply_f(vals[0], vals[1]))
            yield self.pad(result), None
        else:
            yield self.pad(idk), None

    def answer(self, Q, fast_db):
        #TODO: This is very slow, should vectorize
        Q = tuple(Q)
        f_A = self.answer_f_question(Q, fast_db['f'])
        if f_A is not None: return f_A
        x = Q[1:1+self.length]
        if Q[0] in self.simple_question_tokens:
            if x not in fast_db["assignments"]:
                return self.pad(idk)
            result = fast_db["assignments"][x]
            if fast_db["depths"][x] == 0:
                return self.pad(result)
            else:
                if Q[0] == self.simple_query_a:
                    return self.pad(result[0])
                elif Q[0] == self.simple_query_b:
                    return self.pad(result[1])
                assert False
        elif Q[0] == self.compound_query and x in fast_db["values"]:
            return self.pad(fast_db["values"][x])
        else:
            return self.pad(idk)

    def all_questions(self, fast_db):
        for x in fast_db["values"]: yield [self.compound_query, x]

    def is_f_simple(self, Q):
        return self.are_f_simple(np.stack([Q], axis=0))[0]

    def are_simple(self, Qs):
        return np.logical_or(super().are_simple(Qs), self.are_f_simple(Qs))

    def classify_question(self, Q, fast_db):
        Q = tuple(Q)
        if Q[0] in self.simple_question_tokens:
            return "simple"
        if self.is_f_simple(Q):
            return "f"
        return fast_db["depths"][Q[1:1+self.length]]

class EvalSumTask(BaseEvalTask):
    f_interaction_length = 0
    f_question_length = 0
    f_answer_length = 0
    f_fact_length = 0

    def make_values(self, modulus=8):
        self.modulus = modulus
        if modulus is None:
            self.max_d = 2*int(np.sqrt(self.size))
            self.numbers = self.allocate(2 * self.max_d + 1)
            self.zero = self.numbers[self.max_d]
        else:
            self.numbers = self.allocate(self.modulus)
            self.zero = self.numbers[0]

    def apply_f(self, a, b, f_db):
        return self.encode_n(a + b - 2 * self.zero), []

    def encode_n(self, x):
        if self.modulus is None:
            return np.minimum(np.maximum(x, -self.max_d), self.max_d) + self.zero
        else:
            return self.zero + np.mod(x, self.modulus)

    def sample_val(self, f_db):
        return self.encode_n(random.choice([-1, +1]))

    def answer_f_question(self, Q, f_db):
        return None
    
    def repr_f_symbol(self, x):
        if x in self.numbers:
            return str(x - self.zero)

    def recursive_apply_f(self, a, b):
        if False: yield
        a = a[0]
        b = b[0]
        if a not in self.numbers or b not in self.numbers:
            return idk
        return self.encode_n((a - self.zero) + (b - self.zero))

    def make_f_db(self):
        return {}, []

    def are_f_simple(self, Qs):
        nq = Qs.shape[0]
        return np.zeros((nq,), dtype=np.bool)

class EvalTask(BaseEvalTask):
    f_interaction_length = 1
    f_answer_length = 1
    f_question_length = 2
    f_fact_length = 3

    def make_values(self):
        self.num_vals = int(np.sqrt(self.size))
        self.vals = self.allocate(self.num_vals)

    def apply_f(self, a, b, f_db):
        if (a, b) not in f_db:
            c = np.random.choice(self.vals)
            f_db[(a, b)] = c
            return c, [(a, b, c)]
        return f_db[(a, b)], [()]

    def sample_val(self, f_db):
        return np.random.choice(self.vals)

    def answer_f_question(self, Q, f_db):
        Q = (Q[0], Q[1])
        if Q in f_db: return self.pad(f_db[Q])

    def repr_f_symbol(self, x):
        if x in self.vals: return "x{}".format(x)

    def recursive_apply_f(self, a, b):
        a = a[0]
        b = b[0]
        if a not in self.vals or b not in self.vals:
            return idk
        result = (yield None, self.pad((a, b), self.question_length))[0]
        if result not in self.vals:
            return idk
        return result

    def make_f_db(self):
        return {}, []

    def are_f_simple(self, Qs):
        return np.logical_and(np.isin(Qs[:,0], self.vals), np.isin(Qs[:,1], self.vals))
