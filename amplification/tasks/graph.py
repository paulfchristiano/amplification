import random
from collections import defaultdict

import numpy as np

from amplification.tasks.core import idk, uniform, Task, sequences, lexless

#2**log(x) >= x
def log(x):
    return int((np.log(x - 0.5)) / np.log(2)) + 1

def add_logs(l1, l2, powers):
    if l1 not in powers or l2 not in powers: return powers[-1]
    return min(max(l1, l2) + 1, powers[-1])

class GraphTask(Task):
    distance_query_symbol = 1
    step_query_symbol = 2
    neighbor_query_symbol = 3
    edge_query_symbol = 4
    simple_question_tokens = [3, 4]
    question_tokens = [1, 2, 3, 4]
    fixed_vocab = 5
    one = 6
    interaction_length = 9

    def repr_symbol(self, x):
        if x in self.distances:
            if x == self.inf: return '!'
            return str(x - self.zero)
        if x in self.chars:
            return "abcdefghijklmnopqrstuvwxyz"[x - self.chars[0]]
        return {idk:"?",
                self.distance_query_symbol:"D",
                self.step_query_symbol:"S",
                self.neighbor_query_symbol:"N",
                self.edge_query_symbol:"E"}[x]

    def __init__(self, nchars=8, length=2):
        self.size = nchars ** length
        self.length = length
        self.nchars = nchars
        self.nvocab = self.fixed_vocab
        self.max_d = 2 * log(self.size)
        self.distances = self.allocate(self.max_d + 2)
        self.zero = self.distances[0]
        self.inf = self.distances[-1]
        self.chars = self.allocate(self.nchars)
        self.char_powers = nchars ** np.flip(np.arange(self.length), axis=0)
        self.min_char = self.chars[0]
        self.max_char = self.chars[-1]
        self.unindex = np.array([(idk,) * length] + list(sequences(self.chars, length)))
        self.vertices = self.unindex[1:]
        self.question_length = 1 + 2 * length
        self.fact_length = 2 * length
        self.answer_length = length

    def distance_query(self, a, b):
        return self.pad((self.distance_query_symbol,) + tuple(a) + tuple(b), self.question_length)

    def step_query(self, a, b):
        return self.pad((self.step_query_symbol,) + tuple(a) + tuple(b), self.question_length)

    def edge_query(self, a, b):
        return self.pad((self.edge_query_symbol,) + tuple(a) + tuple(b), self.question_length)

    def neighbor_query(self, a):
        return self.pad((self.neighbor_query_symbol,) + tuple(a), self.question_length)

    def encode_n(self, n):
        return np.minimum(self.zero + n, self.inf)

    def add_one_dist(self, d):
        for d0, d1 in zip(self.distances[:-1], self.distances[1:]):
            if d0 == d: return d1
        return self.inf

    def are_chars(self, a):
        return np.logical_and(np.all(a >= self.min_char, axis=-1), np.all(a <= self.max_char, axis=-1))

    def split_questions(self, Qs):
        qs = Qs[:,0]
        valid = np.logical_or(
                np.logical_and(
                    qs == self.neighbor_query_symbol,
                    self.are_chars(Qs[:,1:1+self.length]),
                ),
                np.logical_and(
                    np.isin(qs, self.question_tokens),
                    self.are_chars(Qs[:, 1:]),
                )
            )
        qs = np.where(valid, qs, idk)
        xs = Qs[:,1:1+self.length]
        ys = Qs[:,1+self.length:]
        return qs, xs, ys

    def split_question(self, Q):
        q = Q[0]
        if (q == self.neighbor_query_symbol):
            Q = tuple(Q)
            a = Q[1:1+self.length]
            b = Q[1+self.length:]
            if all(x == idk for x in b) and self.are_chars(a):
                return q, a, b
        elif q in self.question_tokens:
            if self.are_chars(Q[1:]):
                Q = tuple(Q)
                a = Q[1:1+self.length]
                b = Q[1+self.length:]
                return q, a, b
        return None, None, None
    
    def recursive_answer(self, Q):
        q, a, b = self.split_question(Q)
        if q == self.distance_query_symbol:
            is_edge = (yield None, self.edge_query(a, b))
            if is_edge[0] == self.one:
                yield self.pad(self.one), None
                return
            c = (yield None, self.step_query(a, b))
            if np.all(np.isin(c, self.chars)):
                d = (yield None, self.distance_query(c, b))
                if d[0] in self.distances:
                    yield self.pad(self.add_one_dist(d[0])), None
                else:
                    yield self.pad(idk), None
            else:
                yield self.pad(self.inf), None
        elif q == self.step_query_symbol:
            is_edge = (yield None, self.edge_query(a, b))
            if is_edge[0] == self.one:
                yield self.pad(b), None
                return
            def test_vertex(x):
                return self.one == (yield None, self.edge_query(a, x))[0]
            def get_neighbor():
                return (yield None, self.neighbor_query(a))
            def get_step():
                x = (yield None, self.step_query(a, b))
                if self.are_chars(x) and (yield from test_vertex(x)):
                    return x
                return None
            candidates = []
            def addc(x):
                if x is not None and self.are_chars(x): candidates.append(x)
            addc((yield from get_step()))
            addc((yield from get_step()))
            addc((yield from get_neighbor()))
            best = (self.inf,) * self.answer_length
            best_d = self.inf
            for x in candidates:
                d = (yield None, self.distance_query(x, b))
                if d[0] in self.distances and d[0] < self.inf:
                    if (d[0] < best_d or (d[0] == best_d and lexless(x, d))):
                        best_d = d[0]
                        best = x
            yield self.pad(best), None
        else:
            yield self.pad(idk), None

    def make_dbs(self, difficulty=float('inf')):
        num_used_vars = min(8 + difficulty, self.size)
        used_indices = np.random.choice(len(self.vertices), num_used_vars, replace=False)
        used_vars = self.vertices[used_indices]
        num_edges = 2 * num_used_vars
        neighbors = defaultdict(list)
        edges = used_vars[np.random.choice(num_used_vars, (num_edges, 2))]
        edge_indices = self.indices(edges)
        facts = np.concatenate([edges[:,0], edges[:,1]], axis=-1)
        distances = np.ones((self.size + 1, self.size + 1), dtype=np.int32) * 2 * self.size
        #the 0, 0 entry is a placeholder, so that indexing with idk doesn't cause trouble
        distances[edge_indices[:,0], edge_indices[:,1]] = 1
        for k in range(1, self.size + 1):
            np.minimum(distances, distances[:,k, np.newaxis] + distances[np.newaxis, k,:],
                       out=distances)
        fast_db = {"vertices":np.array(used_vars), "distances":distances}
        return facts, fast_db

    def compute_distances(self, distances):
        return distances

    def classify_question(self, Q, fast_db):
        q, a, b = self.split_question(Q)
        t = self.repr_symbol(q)
        if q in self.simple_question_tokens and q != self.edge_query_symbol:
            return t
        a = self.indices(a)
        b = self.indices(b)
        d = fast_db["distances"][a, b]
        if q == self.edge_query_symbol:
            if d == 1:
                return "edge"
            else:
                return "no-edge"
        if d > self.size:
            d = "inf"
        elif d <= 2:
            d = str(d)
        elif d <= 4:
            d = "3-4"
        elif d <= 8:
            d = "5-8"
        elif d <= 16:
            d = "9-16"
        else:
            d = "long"
        return "{}{}".format(t, d)

    def make_qs(self, nqs, fast_db):
        indices = [np.random.randint(len(fast_db["vertices"]), size=(nqs,), dtype=np.int32)
                   for _ in range(2)]
        vertices = [fast_db["vertices"][index] for index in indices]
        Qs = np.random.choice([self.step_query_symbol, self.distance_query_symbol], (nqs, 1))
        return np.concatenate([Qs] + vertices, axis=-1).astype(np.int32)
        

    def indices(self, vs):
        return np.where(self.are_chars(vs),
                        1 + np.sum((vs - self.min_char) * self.char_powers, axis=-1), 0)

    def unindices(self, indices):
        return self.unindex[indices]


    def all_questions(self, fast_db):
        for Q in [self.step_query, self.distance_query]:
            for a in fast_db["vertices"]:
                for b in fast_db["vertices"]:
                        yield Q(a, b)

    def answers(self, Qs, fast_db):
        def answer_edge_queries(xs, ys):
            is_edge = distances[xs, ys] == 1
            no_edge = np.array([[idk, idk]], dtype=np.int32)
            yes_edge = np.array([[self.one, idk]], dtype=np.int32)
            return np.where(np.expand_dims(is_edge, 1), yes_edge, no_edge)
        def answer_neighbor_queries(xs):
            edges = (distances[xs] == 1).astype(np.float32)
            edges[:,0] += 0.1
            noise = 0.01 * np.random.random(edges.shape)
            return self.unindices(np.argmax(noise + edges, axis=1))
        def answer_step_queries(xs, ys):
            is_neighbor = distances[xs, :] <= 1
            is_closer = distances[:, ys].transpose() < np.expand_dims(distances[xs, ys], 1)
            is_closer[np.arange(len(ys)),ys] = 1
            result_indices = np.argmax(np.logical_and(is_neighbor, is_closer), axis=-1)
            result = self.unindices(result_indices)
            return np.where(np.expand_dims(distances[xs, ys], -1) <= self.max_d + 1,
                            result, self.inf)
        def answer_distance_queries(xs, ys):
            d = distances[xs, ys]
            return np.stack([self.encode_n(d), np.zeros_like(d)], axis=-1)

        qs, xs, ys = self.split_questions(Qs)
        distances = fast_db["distances"]
        xs = self.indices(xs)
        ys = self.indices(ys)
        As = np.zeros((Qs.shape[0], self.answer_length), dtype=np.int32)
        b = (qs == self.edge_query_symbol)
        As[b] = answer_edge_queries(xs[b], ys[b])
        b = (qs == self.neighbor_query_symbol)
        As[b] = answer_neighbor_queries(xs[b])
        b = (qs == self.step_query_symbol)
        As[b] = answer_step_queries(xs[b], ys[b])
        b = (qs == self.distance_query_symbol)
        As[b] = answer_distance_queries(xs[b], ys[b])
        return As

#currently decommissioned because it uses answers of length 1
class MidpointTask(GraphTask):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
"""
class MidpointTask(GraphTask):
    distance_query = 1
    midpoint_query = 2
    edge_query = 3
    sample_vertex_query = 4
    used_query = 5
    simple_question_tokens = [edge_query, sample_vertex_query, used_query]
    fixed_vocab = 6
    one = 6

    question_length = 3
    fact_length = 2
    interaction_length = 8

    def repr_symbol(self, x):
        if x in self.powers:
            if x == self.inf: return 'inf'
            return str(2**(x - self.one))
        if x in self.vertices:
            return "v{}".format(x)
        return {idk:"?", self.distance_query:"d", self.midpoint_query:"m",
                self.edge_query:"e", self.sample_vertex_query:"s",
                self.used_query:"u"}[x]

    def __init__(self, size=8):
        self.size = size
        self.num_powers = log(size) + 2 # {2^0, 2^1,... 2^log(size), inf}
        self.nvocab = self.fixed_vocab + self.num_powers + self.size
        self.powers = np.arange(self.fixed_vocab, self.fixed_vocab + self.num_powers)
        assert self.one == self.powers[0]
        self.inf = self.powers[-1]
        self.vertices = np.arange(self.fixed_vocab + self.num_powers, self.nvocab)
    
    def recursive_answer(self, Q):
        if Q[0] == self.distance_query:
            a = Q[1]
            b = Q[2]
            if a not in self.vertices or b not in self.vertices:
                yield idk, None,
                return
            simple_d = (yield None, [self.edge_query, a, b])
            if simple_d == self.one:
                yield self.one, None
                return
            c = (yield None, [self.midpoint_query, a, b])
            if c not in self.vertices:
                yield self.inf, None
                return
            d1 = (yield None, [self.distance_query, a, c])
            d2 = (yield None, [self.distance_query, c, b])
            if d1 in self.powers and d2 in self.powers:
                yield add_logs(d1, d2, self.powers), None
            else:
                yield idk, None
        elif Q[0] == self.midpoint_query:
            a = Q[1]
            b = Q[2]
            if a not in self.vertices or b not in self.vertices:
                yield idk, None,
                return
            def test_vertex(x):
                return self.one == (yield None, [self.used_query, x, idk])
            def get_random_vertex():
                return (yield None, [self.sample_vertex_query, idk, idk])
            def get_midpoint():
                x = (yield None, [self.midpoint_query, a, b])
                if x == idk or not (yield from test_vertex(x)): #short circut is important
                    return None
                return x
            candidates = []
            def addc(x):
                if x in self.vertices: candidates.append(x)
            addc((yield from get_midpoint()))
            if np.random.random() < 0.5:
                addc((yield from get_midpoint()))
            else:
                addc((yield from get_random_vertex()))
            if not candidates: addc((yield from get_random_vertex()))
            best_d = self.inf
            best = idk
            for x in candidates:
                d1 = (yield None, [self.distance_query, a, x])
                d2 = (yield None, [self.distance_query, x, b])
                if d1 in self.powers and d2 in self.powers:
                    if a == x:
                        d = d2
                    elif x == b:
                        d = d1
                    else:
                        d = add_logs(d1, d2, self.powers)
                    if d < best_d:
                        best_d = d
                        best = x
            yield best, None
        else:
            yield idk, None

    def make_q(self, fast_db, simple=False):
        a = np.random.choice(fast_db["vertices"])
        b = np.random.choice(fast_db["vertices"])
        if simple:
            return random.choice([
                [self.edge_query, a, b],
                [self.sample_vertex_query, idk, idk],
                [self.used_query, a, idk]
            ])
        else:
            return [np.random.choice([self.midpoint_query, self.distance_query]), a, b]

    def all_questions(self, fast_db):
        for Q in [self.midpoint_query, self.distance_query]:
            for a in fast_db["vertices"]:
                for b in fast_db["vertices"]:
                        yield [Q, a, b]

    def answer(self, Q, fast_db):
        a = Q[1]
        b = Q[2]
        if Q[0] == self.edge_query:
            if tuple(b) in fast_db["neighbors"].get(a, []):
                return self.one_hot(self.one)
            else:
                return self.one_hot(idk)
        elif Q[0] == self.sample_vertex_query:
            return self.ones(fast_db["vertices"])
        elif Q[0] == self.used_query:
            if a in fast_db["vertices"]:
                return self.one_hot(self.one)
            else:
                return self.one_hot(idk)
        if a not in self.vertices or b not in self.vertices:
            return self.one_hot(idk)
        v0 = self.vertices[0]
        a -= v0
        b -= v0
        distances = fast_db["distances"]
        if Q[0] == self.midpoint_query:
            A = np.zeros(self.nvocab, dtype=np.float32)
            if distances[a, b] == 1:
                A[a + v0] = A[b + v0] = 1
            else:
                cutoff = 2 ** (log(distances[a, b]) - 1)
                A[self.vertices] = np.logical_and(distances[a,:] <= cutoff, distances[:,b] <= cutoff)
                if np.sum(A) == 0: A[idk] = 1
            return A
        elif Q[0] == self.distance_query:
            return self.one_hot(self.one + log(distances[a, b]))
        else:
            return self.one_hot(idk)
"""
