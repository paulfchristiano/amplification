from collections import defaultdict
import numpy as np
import random

from amplification.tasks.core import idk, Task, sequences

#yields edges of a random tree on [a, b)
#if b = a+1, yields nothing
#if point to is not none, all edges (x, y) have y closer to point_to
def random_tree(a, b, point_to=None):
    if a + 1 < b:
        split = np.random.randint(a+1, b)
        l = np.random.randint(a, split)
        r = np.random.randint(split, b)
        if point_to is None:
            yield (l, r)
            yield from random_tree(a, split)
            yield from random_tree(split, b)
        else:
            point_left = point_to < split
            yield (r, l) if point_left else (l, r)
            yield from random_tree(a, split, point_to if point_left else l)
            yield from random_tree(split, b, r if point_left else point_to)
    elif a >= b:
        raise ValueError()

def dfs(neighbors_dict, start_node):
    depths = {start_node: 0}
    parents = {start_node: start_node}

    stack = [start_node]

    while stack:
        node = stack.pop()

        children = neighbors_dict[node]
        for child in children:
            if child not in depths:
                stack.append(child)
                depths[child] = depths[node] + 1
                parents[child] = node

    return parents, depths


class EqualsTask(Task):
    value_query = 1
    simple_value_query = 2
    neighbor_query = 3
    parent_query = 4
    depth_query = 5
    fixed_vocab = 6

    interaction_length = 9

    simple_question_tokens = [simple_value_query, neighbor_query]

    def repr_symbol(self, x):
        if x in self.chars: return "abcdefghijklmnopqrstuv"[x]
        if x in self.vals: return str(x)
        if x in self.depths: return str(x - self.zero)
        return {self.value_query: "V",
                self.neighbor_query: "N",
                self.simple_value_query: "S",
                self.parent_query: "P",
                self.depth_query: "D"}.get(x, "?")

    def __init__(self, nchars=8, length=2, num_vals=None, easy=False):
        self.nchars = nchars
        self.length = length
        self.num_vars = nchars ** length
        self.num_vals = num_vals or int(np.sqrt(self.num_vars))
        self.nvocab = self.fixed_vocab
        self.chars = self.allocate(self.nchars)
        self.min_char = self.chars[0]
        self.max_char = self.chars[-1]
        self.vars = list(sequences(self.chars, self.length))
        self.vals = self.allocate(self.num_vals)
        self.max_d = self.num_vars - self.num_vals
        self.depths = self.allocate(self.max_d + 1)
        self.zero = self.depths[0]
        self.largest_d = self.depths[-1]
        self.easy = easy
        self.fact_length = 2 * length
        self.answer_length = length
        self.question_length = 2 * length

    def encode_n(self, n):
        return np.minimum(self.zero + n, self.largest_d)

    def are_simple(self, Qs):
        return np.logical_or(
            np.isin(Qs[:,0], self.simple_question_tokens),
            np.logical_and(
                np.isin(Qs[:,0], self.vars),
                np.isin(Qs[:,1], self.vars)
            )
        )

    def make_simple_value_query(self, x):
        return self.pad((self.simple_value_query,) + tuple(x), self.question_length)

    def make_parent_query(self, x):
        return self.pad((self.parent_query,) + tuple(x), self.question_length)

    def make_value_query(self, x):
        return self.pad((self.value_query,) + tuple(x), self.question_length)

    def make_neighbor_query(self, x):
        return self.pad((self.neighbor_query,) + tuple(x), self.question_length)

    def make_depth_query(self, x):
        return self.pad((self.depth_query,) + tuple(x), self.question_length)

    def make_edge_query(self, x, y):
        return self.pad(tuple(x) + tuple(y), self.question_length)

    def are_chars(self, x):
        return np.logical_and(np.all(x >= self.min_char, axis=-1), np.all(x <= self.max_char, axis=-1))

    def recursive_answer(self, Q):
        Q = tuple(Q)
        x = Q[1:1+self.length]
        if Q[0] == self.value_query:
            if not self.are_chars(x):
                yield self.pad(idk), None
                return
            simple_value = (yield None, self.make_simple_value_query(x))[0]
            if simple_value in self.vals:
                yield self.pad(simple_value), None
                return
            y = (yield None, self.make_parent_query(x))[:self.length]
            if not self.are_chars(y):
                yield self.pad(idk), None
                return
            val = (yield None, self.make_value_query(y))[0]
            if val not in self.vals:
                yield self.pad(idk), None
                return
            yield self.pad(val), None
        elif Q[0] == self.depth_query:
            if not self.are_chars(x):
                yield self.pad(idk), None
                return
            simple_val = (yield None, self.make_simple_value_query(x))[0]
            if simple_val in self.vals:
                yield self.pad(self.zero), None
                return
            y = (yield None, self.make_parent_query(x))[:self.length]
            if not self.are_chars(y):
                yield self.pad(idk), None
                return
            d = (yield None, self.make_depth_query(y))[0]
            if d not in self.depths:
                yield self.pad(idk), None
                return
            yield self.pad(self.encode_n(d - self.zero + 1)), None
        elif Q[0] == self.parent_query:
            if not self.are_chars(x):
                yield idk, None
                return
            simple_val = (yield None, self.make_simple_value_query(x))[0]
            if simple_val in self.vals:
                yield x, None
                return
            def test_var(y): return self.zero == (yield None, self.make_edge_query(x, y))[0]
            def get_neighbor(): return (yield None, self.make_neighbor_query(x))[:self.length]
            def get_parent():
                y = (yield None, self.make_parent_query(x))[:self.length]
                if self.are_chars(y) and (yield from test_var(y)):
                    return y
                return None
            candidates = []
            def addc(y):
                if y is not None and self.are_chars(y): candidates.append(y)
            addc((yield from get_parent()))
            addc((yield from get_parent()))
            addc((yield from get_neighbor()))
            best = idk
            best_d = self.largest_d
            for y in candidates:
                d = (yield None, self.make_depth_query(y))[0]
                if d in self.depths and d<= best_d:
                    best_d = d
                    best = y
            yield self.pad(best), None
        else:
            yield self.pad(idk), None

    def make_dbs(self, difficulty=float('inf')):
        difficulty = min(self.num_vars, difficulty)
        num_used_vars = min(difficulty + 10, self.num_vars)
        num_used_vals = min(int(np.sqrt(difficulty+10)), self.num_vals)
        used_vars = random.sample(self.vars, num_used_vars)
        used_vals = np.random.choice(self.vals, num_used_vals, replace=False)
        value_list = np.random.choice(used_vals, num_used_vars, replace=True)
        state = {}
        given_values = {}
        given_equivalences = []
        neighbors = defaultdict(list)
        equivalence_classes = defaultdict(list)
        for var, val in zip(used_vars, value_list):
            equivalence_classes[val].append(var)
            state[var] = val
        for val, vs in equivalence_classes.items():
            root = random.choice(vs)
            given_values[root] = val
            tree = random_tree(0, len(vs), point_to=root if self.easy else None)
            given_equivalences.extend([(vs[i], vs[j]) for i, j in tree])
        for x, y in given_equivalences:
            neighbors[x].append(y)
            neighbors[y].append(x)
        facts = np.array(
            [self.pad(tuple(var) + (val,), self.fact_length) for var, val in given_values.items()] +
            [self.pad(tuple(var1) + tuple(var2), self.fact_length) for var1, var2 in given_equivalences])

        parents = {}
        depths = {}
        for y in given_values:
            new_parents, new_depths = dfs(neighbors, y)
            parents.update(new_parents)
            depths.update(new_depths)

        fast_db = {"givens": given_values, "neighbors": neighbors, "values": state, "inverse": equivalence_classes,
                   'depths': depths, 'parents':parents,
                   "used_vars":used_vars}

        return facts, fast_db

    def classify_question(self, Q, fast_db):
        Q = tuple(Q)
        t = self.repr_symbol(Q[0])
        if Q[0] in self.simple_question_tokens:
            return t
        return "{}{}".format(t, fast_db['depths'][Q[1:1+self.length]])

    def make_q(self, fast_db):
        q = random.choice([self.value_query, self.parent_query, self.depth_query])
        v = random.choice(fast_db["used_vars"])
        return self.pad((q,) + v, self.question_length)

    def answer(self, Q, fast_db):
        Q = tuple(Q)
        x = Q[1:1+self.length]
        if Q[0] == self.simple_value_query:
            return self.pad(fast_db["givens"].get(x, idk))
        elif Q[0] == self.neighbor_query:
            neighbors = fast_db["neighbors"].get(x, [])
            if neighbors:
                return self.pad(random.choice(neighbors))
            else:
                return self.pad(idk)
        elif Q[0] == self.depth_query:
            if x in fast_db["depths"]:
                return self.pad(self.encode_n(fast_db["depths"][x]))
            else:
                return self.pad(idk)
        elif Q[0] == self.parent_query:
            return self.pad(fast_db["parents"].get(x, idk))
        elif Q[0] == self.value_query:
            return self.pad(fast_db["values"].get(x, idk))
        else:
            if Q[:self.length] in fast_db["neighbors"]:
                if Q[self.length:2*self.length] in fast_db["neighbors"][Q[:self.length]]:
                    return self.pad(self.zero)
            return self.pad(idk)

    def all_questions(self, fast_db):
        for var in self.vars:
            yield [self.value_query, var]
