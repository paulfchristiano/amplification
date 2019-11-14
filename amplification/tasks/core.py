import numpy as np

idk = 0

def uniform(x):
    return np.random.choice(x) if x else idk

#returns: (x|x!=idk), p(x!=idk), whether the conditioning failed
def is_known(x):
    p = 1 - x[idk]
    if p < 1e-9:
        return 0*x, 0, True
    x[idk] = 0
    x = x / p
    return x, p, False

def sample(x, peak=False):
    if peak:
        return np.argmax(x)
    else:
        psum = np.sum(x)
        return np.random.choice(len(x), p=x/psum)

def lexless(a, b):
    return np.any(a < b) and (np.all(a <= b) or (np.argmin(a < b) < np.argmin(b < a)))

def sample_known(x):
    x, p, failed = is_known(x)
    return None if failed else sample(x)

def sequences(alphabet, length):
    yield from combos([alphabet] * length)

def combos(alphabets):
    if alphabets:
        for x in alphabets[0]:
            for xs in combos(alphabets[1:]):
                yield (x,) + xs
    else:
        yield ()

class Task():
    interaction_length = 10

    def get_batch(self, nbatch, nqs=None, **kwargs):
        facts, fast_dbs, Qs, As = [], [], [], []
        for batchn in range(nbatch):
            fact, fast_db = self.make_dbs(**kwargs)
            if batchn == 0: nfacts = len(fact)
            assert nfacts == len(fact)
            if nqs is None: nqs = nfacts
            facts.append(fact)
            fast_dbs.append(fast_db)
            Q = self.make_qs(nqs, fast_db)
            A = self.answers(Q, fast_db)
            Qs.append(Q)
            As.append(A)

        return (np.array(facts), fast_dbs, np.array(Qs), np.array(As))

    def allocate(self, n):
        result = np.arange(self.nvocab, self.nvocab + n)
        self.nvocab += n
        return result

    @property
    def transcript_length(self):
        #each round has length = question_length + answer_length
        #one round per interaction_length, plus the starting question and final answer
        return (self.interaction_length + 1) * (self.question_length + self.answer_length)

    def repr_symbols(self, xs):
        return " ".join([self.repr_symbol(x) for x in xs])

    def repr_answer(self, ans):
        return self.repr_symbols(ans)

    def repr_fact(self, fact):
        return self.repr_symbols(fact)

    def repr_question(self, q):
        return self.repr_symbols(q)

    def pad(self, a, pad_to=None):
        if pad_to is None: pad_to = self.answer_length
        try:
            a = tuple(a)
        except TypeError:
            a = (a,)
        return a + (idk,) * (pad_to - len(a))

    def ones(self, xs):
        result = np.zeros(self.nvocab, dtype=np.float32)
        for x in xs: result[x] = 1
        if np.sum(result) < 1: result[idk] = 1
        return result

    def uniform(self, xs):
        result = np.zeros(self.nvocab, dtype=np.float32)
        k = len(xs)
        if k == 0:
            result[idk] = 1
        else:
            for x in xs: result[x] = 1/k
        return result

    def delta(self, x):
        return self.uniform([x])
    
    def one_hot(self, x):
        return np.eye(self.nvocab)[np.asarray(x)]

    def classify_question(self, fast_db, Q):
        return "all"

    def are_simple(self, Qs):
        return np.isin(Qs[:,0], self.simple_question_tokens)

    def is_simple(self, Q):
        return self.are_simple(np.stack([Q], axis=0))[0]

    def make_qs(self, nqs, fast_db):
        return np.array([self.make_q(fast_db) for _ in range(nqs)])

    def answers(self, Qs, fast_db):
        result = np.zeros((Qs.shape[0], self.answer_length), np.int32)
        for i, Q in enumerate(Qs):
            result[i] = self.answer(Q, fast_db)
        return result

    def answer(self, Q, fast_db):
        try:
            return self.answers(np.stack([Q], axis=0), fast_db)[0]
        except RecursionError: #happens if we don't implement one of answer or answers
            raise NotImplementedError

def test_task(task, nbatch=10, nqs=300):
    import time
    fast_dbs = []
    t0 = time.time()
    for i in range(nbatch):
        fast_dbs.append(task.make_dbs()[1])
    t = (time.time() - t0) / nbatch
    print("Generate one environment: {:.1E}".format(t))
    t0 = time.time()
    for fast_db in fast_dbs:
        answerer = lambda Qs : task.answers(Qs, fast_db)
        Qs = task.make_qs(nqs, fast_db)
        for Q in Qs:
            task.classify_question(Q, fast_db)
        direct_A = task.answers(Qs, fast_db)
        recursive_A, subQs, subAs = recursive_run(task, Qs, answerer)
        assert np.all(direct_A == recursive_A)
    t = (time.time() - t0) / (nbatch * nqs)
    print("Answer one question: {:.1E}".format(t))

def test_all_tasks(**kwargs):
    from amplification.tasks import EqualsTask, GraphTask, MidpointTask
    from amplification.tasks import SumTask, EvalTask, IterTask, EvalSumTask
    for task in [
            GraphTask(),
            EqualsTask(),
            IterTask(),
            EvalSumTask(),
            EvalSumTask(modulus=None),
            SumTask(),
            SumTask(modulus=None),
            EvalTask(),
            #MidpointTask(),
            ]:
        print("Testing {}".format(type(task).__name__))
        test_task(task, **kwargs)

def pad_with_none(it):
    yield from it
    while True: yield None, None

#Any shape of Qs
def recursive_run(task, Qs, answerer):
    def flatten(x):
        x = np.asarray(x)
        flatten.shape = x.shape[:-1]
        return np.reshape(x, (-1, x.shape[-1]))
    def unflatten(x):
        x = np.asarray(x)
        return np.reshape(x, flatten.shape + x.shape[1:])
    flat_Qs = flatten(Qs)
    def flat_answerer(flat_subQs):
        flat_subQs = np.asarray(flat_subQs)
        subQs = unflatten(flat_subQs)
        subAs = answerer(subQs)
        return flatten(subAs)
    flat_As, flat_subQs, flat_subAs = recursive_run_list(task, flat_Qs, flat_answerer)
    return unflatten(flat_As), unflatten(flat_subQs), unflatten(flat_subAs)

#list of Qs
def recursive_run_list(task, Qs, answerer):
    nqs = len(Qs)
    As = -np.ones((nqs, task.answer_length), dtype=np.int32)
    subQs = np.zeros((nqs, task.interaction_length, task.question_length), dtype=np.int32)
    subAs = np.zeros((nqs, task.interaction_length, task.answer_length), dtype=np.int32)
    askers = [pad_with_none(task.recursive_answer(Q)) for Q in Qs]
    responses = [next(asker) for asker in askers]
    for j in range(task.interaction_length):
        for i, (A, subQ) in enumerate(responses):
            if subQ is not None: subQs[i][j] = subQ
            if A is not None: As[i] = A
        subAs[:,j] = answerer(subQs[:,j])
        responses = [asker.send(subAs[i,j]) for i, asker in enumerate(askers)]
    for i, (A, subQ) in enumerate(responses):
        if A is not None: As[i] = A
    assert np.all(As >= 0)
    return As, subQs, subAs

def print_interaction(task, Q, subQs, subAs, A, fast_db):
    print("{} ({})".format(task.repr_question(Q), task.classify_question(Q, fast_db)))
    for subQ, subA in zip(subQs, subAs):
        if np.any(subQ != idk):
            print("  {}: {}".format(task.repr_question(subQ),
                                    task.repr_answer(subA)))
    print("  A: {}".format(task.repr_answer(A)))

