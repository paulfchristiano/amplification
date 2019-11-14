import random
from collections import defaultdict

import numpy as np

from amplification.tasks.core import idk, uniform, Task, sequences

class IterTask(Task):
    interaction_length = 3

    zero = 1
    one = 2
    fixed_vocab = 3

    def repr_symbol(self, x):
        if x == idk: return '?'
        if x == self.one: return '1'
        if x == self.zero: return '0'
        if x in self.chars: return 'abcdefghijklmnopqrstuv'[x]

    def __init__(self, nchars=8, length=2, log_iters=6):
        self.nvocab = self.fixed_vocab
        self.nchars = nchars
        self.length = length
        self.log_iters = log_iters
        self.size = nchars ** length
        self.chars = self.allocate(nchars)
        self.min_char = self.chars[0]
        self.max_char = self.chars[-1]
        self.vars = list(sequences(self.chars, length))
        self.question_length = length + log_iters
        self.fact_length = 2 * length
        self.answer_length = length

    def make_dbs(self, difficulty=float('inf')):
        size = min(difficulty+8, self.size)
        used_vars = random.sample(self.vars, size)
        vals_raw = np.random.permutation(size)
        vals = np.array(used_vars)[vals_raw]
        square_raw = vals_raw
        squares_raw = [square_raw]
        for i in range(self.log_iters):
            square_raw = square_raw[square_raw]
            squares_raw.append(square_raw)
        squares = [{val:used_vars[squares_raw[i][index]] for index, val in enumerate(used_vars)}
                   for i in range(self.log_iters)]
        fast_db = {"vals": {v:val for v, val in zip(used_vars, vals)},
                   "vars":  used_vars,
                   "squares_raw": squares_raw,
                   "squares": squares}
        facts = np.concatenate([np.array(used_vars), vals], axis=1)
        return facts, fast_db

    def are_chars(self, x):
        return np.logical_and(np.all(x >= self.min_char), np.all(x <= self.max_char))

    def recursive_answer(self, Q):
        Q = tuple(Q)
        x = Q[:self.length]
        n = Q[self.length:]
        if not self.are_chars(x) or not np.all(np.isin(n, [self.zero, self.one])):
            yield self.pad(idk), None
            return
        if np.all(n[:-1] == self.zero):
            yield (yield None, Q), None
            return
        leading_bit = np.argmax(n)
        shifted = self.zero * np.ones(self.log_iters, dtype=np.int32)
        shifted[1:] = n[:-1]
        queries = [shifted, shifted]
        if n[-1] == self.one:
            parity = self.zero * np.ones(self.log_iters, dtype=np.int32)
            parity[-1] = self.one
            queries.append(parity)
        def query(x, n): return np.concatenate([x, n])
        for m in queries:
            x = (yield None, query(x, m))
            if not self.are_chars(x):
                yield self.pad(idk), None
                return
        yield self.pad(x), None

    def make_q(self, fast_db):
        x = random.choice(fast_db["vars"])
        n = np.ones(self.log_iters, dtype=np.int32) * self.zero
        leading_bit = np.random.randint(0, self.log_iters-1)
        n[leading_bit] = self.one
        remainder = self.log_iters- leading_bit - 1
        n[leading_bit+1:] = np.random.choice([self.zero, self.one], remainder)
        return np.concatenate([x, n])

    def answer(self, Q, fast_db):
        Q = tuple(Q)
        x = Q[:self.length]
        n = Q[self.length:]
        if x in fast_db["vals"]:
            for i in range(self.log_iters):
                if n[i] == self.one:
                    x = fast_db["squares"][self.log_iters - i - 1][x]
                elif n[i] == self.zero:
                    x = x
                else:
                    return self.pad(idk)
            return self.pad(x)
        else:
            return self.pad(idk)

    def all_questions(self, fast_db):
        for x in fast_db["vars"]:
            for n in sequences([self.zero, self.one], self.log_iters):
                yield np.asarray([x] + n)

    def are_simple(self, Q):
        return np.all(Q[:,self.length:-1] == self.zero, axis=-1)

    def classify_question(self, Q, fast_db):
        n = Q[self.length:]
        if np.all(n == self.zero):
            return 0
        leading_bit = np.argmax(n)
        return self.log_iters - leading_bit
