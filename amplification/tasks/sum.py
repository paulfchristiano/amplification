import random
from collections import defaultdict

import numpy as np

from amplification.tasks.core import idk, uniform, Task, sequences

def matches(patterns, xs):
    return np.all((patterns[:,np.newaxis,:] == SumTask.wild) |
                  (patterns[:,np.newaxis,:] == xs[np.newaxis, :, :]), axis=2)

class SumTask(Task):
    wild = 1
    answer_length = 1
    fixed_vocab = 2

    def repr_symbol(self, x):
        if x == self.wild: return '*'
        if x in self.differences:
            return str(x - self.zero)
        if x in self.alphabet:
            return 'abcdef'[x - self.alphabet[0]]
        if x == idk: return '?'
        raise ValueError(x)

    def __init__(self, length=6, size=float('inf'), nchars=2, modulus=None):
        self.nvocab = self.fixed_vocab
        self.size = min(size, nchars**length)
        self.alphabet = self.allocate(nchars)
        self.interaction_length = nchars
        self.alphabet_plus = np.concatenate([[self.wild], self.alphabet])
        self.modulus = modulus
        if modulus is None:
            self.max_d = (self.size + nchars - 1) // nchars
            self.differences = self.allocate(2 * self.max_d + 1)
            self.zero = self.differences[self.max_d]
        else:
            self.differences = self.allocate(self.modulus)
            self.zero = self.differences[0]
        self.all_strings = [np.array(x) for x in sequences(self.alphabet, length)]
        self.length = length
        self.question_length = length
        self.fact_length = length + 1

    def make_dbs(self, difficulty=float('inf')):
        used_strings = min(self.size, difficulty+8)
        strings = np.stack(random.sample(self.all_strings, used_strings))
        values = np.random.choice([-1, 1], used_strings)
        fast_db = {"strings": strings, "values": values}
        facts = np.concatenate([strings, self.encode_n(values[:,np.newaxis])], axis=1)
        return facts, fast_db

    def answers(self, Qs, fast_db):
        all_matches = matches(Qs, fast_db["strings"])
        raw_As = np.dot(all_matches, fast_db["values"])
        As = self.encode_n(raw_As)
        return As[:, np.newaxis]

    def make_q(self, fast_db):
        Q = np.random.choice(self.alphabet, self.length, replace=True)
        num_wilds = np.random.randint(1, self.length + 1)
        indices = np.random.choice(self.length, num_wilds, replace=False)
        Q[indices] = self.wild
        return Q

    def encode_n(self, x):
        if self.modulus is None:
            return self.zero + np.maximum(-self.max_d, np.minimum(self.max_d, x))
        else:
            return self.zero + np.mod(x, self.modulus)

    def are_simple(self, Qs):
        return np.all(Qs != self.wild, axis=-1)

    def recursive_answer(self, Q):
        Q = np.asarray(Q)
        if not np.all(np.isin(Q, self.alphabet_plus)):
            yield self.pad(self.zero), None
            return
        if not np.any(Q == self.wild):
            yield (yield None, Q), None
            return
        wild_index = np.argmax(Q == self.wild)
        result = 0
        for c in self.alphabet:
            new_Q = np.copy(Q)
            new_Q[wild_index] = c
            d = (yield None, new_Q)
            if d not in self.differences:
                yield self.pad(idk), None
                return
            result += d - self.zero
        result = self.encode_n(result)
        yield self.pad(result), None

    def all_questions(self, fast_db):
        yield from sequences(self.alphabet_plus, self.length)

    def classify_question(self, Q, fast_db):
        n = len([x for x in Q if x == self.wild])
        return "wilds{}".format(n)
