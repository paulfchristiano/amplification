import time
import threading
import sys
import itertools
from collections import defaultdict

import tensorflow as tf
import numpy as np

import amplification.models as models
from amplification.tasks.core import idk, print_interaction, recursive_run
from amplification.buffer import Buffer
from amplification.logger import Logger

print_lock = threading.Lock()

def multi_access(d, ks):
    for k in ks.split("/"):
        d = d[k]
    return d

def get_accuracy(guesses, ground_truth):
    return np.mean(np.all(guesses == ground_truth, axis=-1))

class Averager():
    def __init__(self):
        self.reset()
    def add(self, c, x, horizon=None):
        with self.locks[c]:
            if horizon is not None:
                self.n[c] *=  (1 - 1/horizon)
                self.sum[c] *= (1 - 1/horizon)
            self.n[c] += 1
            self.sum[c] += x
    def items(self):
        for c in self.n: yield (c, self.get(c))
    def get(self, c):
        with self.locks[c]:
            return self.sum[c] / self.n[c]
    def reset(self):
        self.n = defaultdict(lambda:1e-9)
        self.sum = defaultdict(lambda:0)
        self.locks = defaultdict(threading.Lock)

def get_interactions(run, task, facts, fast_dbs, Qs,
        use_real_answers=False, use_real_questions=True):

    if not use_real_questions:
        assert not use_real_answers
        return  run(["targets", "subQs", "subAs"], facts=facts, Qs=Qs, fast_dbs=fast_dbs)

    if use_real_answers:
        answerer = lambda Qss: np.array([task.answers(Qs, fast_db)
                                         for Qs, fast_db in zip(Qss, fast_dbs)])
    else:
        answerer = lambda Qss: run(['teacher_or_simple'],
                                   facts=facts, Qs=Qss, fast_dbs=fast_dbs,
                                   is_training=False)[0]

    return recursive_run(task, Qs, answerer)

def print_batch(task, Qs, subQs, subAs, As, facts, fast_dbs, **other_As):
    with print_lock:
        print()
        print()
        print("Facts:")
        for fact in facts[0]:
            print("  " + task.repr_fact(fact))
        for i in range(min(5, len(Qs[0]))):
            print()
            print_interaction(task, Qs[0,i], subQs[0,i], subAs[0,i], As[0,i], fast_dbs[0])
            for k, v in sorted(other_As.items()):
                print("{}: {}".format(k, task.repr_answer(v[0,i])))

def log_accuracy(task, Qs, ground_truth, fast_dbs, stats_averager, **As_by_name):
    classification = np.asarray([[task.classify_question(Q, fast_db) for Q in Q_list]
                                 for fast_db, Q_list in zip(fast_dbs, Qs)])
    total = np.size(classification)
    with print_lock:
        for name, As in As_by_name.items():
            accuracies = np.all(As == ground_truth, axis=-1)
            correct = np.sum(accuracies)
            classes = set(np.reshape(classification, [-1]))
            counts = {c: np.sum(classification == c) for c in classes}
            correct_counts = {c: np.sum((classification == c) * accuracies) for c in classes}
            def repr_accuracy(k, N): return "{}% ({}/{})".format(int(100*k/N), int(k), int(N))
            print()
            print("{} accuracy: {}".format(name, repr_accuracy(correct, total)))
            stats_averager.add("accuracy/{}".format(name), correct/total)
            for c in sorted(counts.keys()):
                print("  {}: {}".format(c, repr_accuracy(correct_counts[c], counts[c])))
                stats_averager.add("accuracy_on/{}/{}".format(c, name), correct_counts[c]/counts[c])

def generate_answerer_data(run, task, get_batch, answerer_buffer, stats_averager, stepper,
        use_real_questions=False, use_real_answers=False):
    averager = Averager()
    while True:
        facts, fast_dbs, Qs, ground_truth = get_batch()
        nqs = Qs.shape[1]
        As, subQs, subAs = get_interactions(run, task, facts, fast_dbs, Qs,
                use_real_answers=use_real_answers,
                use_real_questions=use_real_questions)
        teacher_As, = run(["answerer/teacher/As"],
                                     facts=facts, Qs=Qs, is_training=False)
        teacher_quality = np.mean(np.logical_and(
            np.any(teacher_As != idk, axis=-1),
            np.all(teacher_As == As, axis=-1)
        ))
        log_accuracy(task, Qs, ground_truth, fast_dbs, stats_averager,
                teacher=teacher_As, targets=As)
        print_batch(task, Qs, subQs, subAs, As, facts, fast_dbs,
                    teacher=teacher_As, truth=ground_truth)
        batch = {"facts":facts, "Qs":Qs, "targets":As, "truth":ground_truth}
        answerer_buffer.extend(batch, extendible={x:[1] for x in answerer_buffer.keys()})
        averager.add("quality", teacher_quality, 100)
        stats_averager.add("quality/teacher", teacher_quality)
        if averager.get("quality") > 0.85 and averager.n["quality"] > 50:
            get_batch.difficulty += 1
            averager.reset()
        stepper["answerer_gen"] += 1

def make_validation_buffer(task, instances=1000, nqs=50, min_difficulty=0, max_difficulty=56):
    difficulty_counts = [1] + ([1/(max_difficulty - min_difficulty)] * (max_difficulty - min_difficulty - 1)) + [1]
    difficulty_sum = sum(difficulty_counts)
    difficulty_counts = [int(d * instances / difficulty_sum) for d in difficulty_counts]
    result = Buffer(instances,
        {"facts": [0, task.fact_length],
         "Qs": [0, task.question_length],
         "truth": [0, task.answer_length]},
    )
    for i, n in enumerate(difficulty_counts):
        difficulty = i + min_difficulty
        facts, fast_dbs, Qs, ground_truth = task.get_batch(n, nqs=nqs, difficulty=difficulty)
        batch = {"facts":facts, "Qs":Qs, "truth":ground_truth}
        result.extend(batch, extendible={x:[1] for x in result.keys()})
    return result

def train_answerer(run, answerer_buffer, stats_averager, make_log, stepper, nbatch, task):
    validation_buffer = make_validation_buffer(task)
    while not answerer_buffer.has(10*nbatch):
        time.sleep(0.1)
    while True:
        batch = answerer_buffer.sample(nbatch)
        _, loss, As = run(["answerer/train",
                           "answerer/student/loss",
                           "answerer/student/As"],
                          batch, is_training=True)
        accuracy = get_accuracy(As, batch["truth"])
        stats_averager.add("accuracy/train", accuracy)
        stats_averager.add("loss/answerer", loss)
        stepper["answerer_train"] += 1
        if stepper["answerer_train"] % 5 == 0:
            batch = validation_buffer.sample(nbatch)
            As, = run(["answerer/student/As"], batch, is_training=False)
            accuracy = get_accuracy(As, batch["truth"])
            stats_averager.add("accuracy/validation", accuracy)
        if stepper["answerer_train"] % 10 == 0:
            make_log()

def generate_asker_data(run, task, get_batch, asker_buffer, stats_averager, stepper,
        use_real_answers=False, max_nbatch=50):
    averager = Averager()
    needed_labels = 10 * max_nbatch
    last_polled = 0
    while True:
        current = stepper["answerer_train"]
        elapsed = current - last_polled
        last_polled = current
        if averager.get("loss") < 0.01:
            rate = 0.01
        elif averager.get("loss") < 0.1:
            rate = 0.1
        else:
            rate = 1
        needed_labels += rate * elapsed
        nbatch = int(min(max_nbatch, needed_labels))
        needed_labels -= nbatch
        if nbatch == 0:
            time.sleep(0.1)
        else:
            facts, fast_dbs, Qs, ground_truth = get_batch(nbatch)
            As, subQs, subAs = get_interactions(run, task, facts, fast_dbs, Qs,
                    use_real_answers=use_real_answers,
                    use_real_questions=True)
            all_transcripts = []
            all_tokens = []
            for batchn in range(nbatch):
                #get on one random question per batch
                #(we throw away the others and pretend they never existed)
                qn = np.random.randint(Qs.shape[1])
                transcripts, tokens = models.asker.make_transcript(Qs[batchn, qn],
                                                                   subQs[batchn, qn],
                                                                   subAs[batchn, qn],
                                                                   As[batchn, qn])
                all_transcripts.append(transcripts)
                all_tokens.append(tokens)
            batch = {"transcripts":np.array(all_transcripts),
                     "token_types":np.array(all_tokens)}
            asker_buffer.extend(batch)
            new_loss, = run(["asker/loss"], batch, is_training=False)
            averager.add("loss", new_loss, 3000 * rate/nbatch)
            stats_averager.add("loss/asker/validation", averager.get("loss"))
            stepper["asker_gen"] += nbatch

def train_asker(run, asker_buffer, stats_averager, stepper, nbatch):
    while not asker_buffer.has(5*nbatch):
        time.sleep(1)
    while True:
        if stepper["asker_train"] > stepper["answerer_train"] + 10000:
            time.sleep(0.1)
        else:
            batch = asker_buffer.sample(nbatch)
            _, loss = run(["asker/train", "asker/loss"], batch, is_training=True)
            stats_averager.add("loss/asker", loss)
            stepper["asker_train"] += 1

def train(task, model, nbatch=50, num_steps=400000,
        path=None,
        stub=False, learn_human_model=True, supervised=False, curriculum=True,
        generation_frequency=10, log_frequency=10,
        buffer_size=10000, asker_data_limit=100000, loss_threshold=0.3):
    if supervised: learn_human_model = False
    if not stub:
        placeholders = {
            "facts": tf.placeholder(tf.int32, [None, None, task.fact_length], name="facts"),
            "Qs": tf.placeholder(tf.int32, [None, None, task.question_length], name="Qs"),
            "targets": tf.placeholder(tf.int32, [None, None, task.answer_length],
                                      name="targets"),
            "transcripts": tf.placeholder(tf.int32, [None, None], name="transcripts"),
            "token_types": tf.placeholder(tf.int32, [None, None], name="token_types"),
            'is_training': tf.placeholder(tf.bool, [], name='is_training'),
        }
    answerer_buffer = Buffer(buffer_size,
        {"facts": [0, task.fact_length],
         "Qs": [0, task.question_length],
         "targets": [0, task.answer_length],
         "truth": [0, task.answer_length]},
        validation_fraction=0.1,
    )
    asker_buffer = Buffer(asker_data_limit,
        {"transcripts":[task.transcript_length],
         "token_types":[task.transcript_length]},
        validation_fraction=0.1,
    )
    #keep track of how many times we've performed each kind of step
    stepper = {"answerer_train":0, "asker_train":0, "answerer_gen":0, "asker_gen":0}

    #this is machinery for passing python objects into sess.run
    #you actually pass in the index of the object in fast_db_communicator,
    #and wrapped python functions use that
    fast_db_communicator = {'next':0}
    fast_db_index = tf.placeholder(tf.int32, [], name="fast_db_index")

    def answer_if_simple_py(fast_db_index, Qs):
        fast_dbs = fast_db_communicator[fast_db_index]       
        As = np.zeros(Qs.shape[:-1] + (task.answer_length,), np.int32)
        are_simple = np.zeros(Qs.shape[:-1], np.bool)
        for i, fast_db in enumerate(fast_dbs):
            are_simple[i] = task.are_simple(Qs[i])
            As[i, are_simple[i]] = task.answers(Qs[i, are_simple[i]], fast_db)
        return are_simple, As

    def answer_if_simple_tf(Qs):
        return tf.py_func(answer_if_simple_py, [fast_db_index, Qs], (tf.bool, tf.int32))

    def make_feed(d):
        result = {}
        cleanup = lambda : None
        for k, v in d.items():
            if k in placeholders:
                result[placeholders[k]] = v
            if k == "fast_dbs":
                fast_db_communicator["next"] += 1
                next_index = fast_db_communicator["next"]
                result[fast_db_index] = next_index
                fast_db_communicator[next_index] = v
                def cleanup(): del fast_db_communicator[next_index]
        return result, cleanup

    def run(op_names, batch={}, **kwargs):
        kwargs.update(batch)
        if not stub:
            to_run = [multi_access(ops, op_name) for op_name in op_names]
            feed_dict, cleanup = make_feed(kwargs)
            try:
                return sess.run(to_run, feed_dict)
            finally:
                cleanup()
        if stub:
            def As(batch):
                return np.zeros(batch["Qs"].shape[:2] + (task.answer_length,), dtype=np.int32)
            def subQs(batch):
                return np.zeros(batch["Qs"].shape[:2] +
                            (task.interaction_length, task.question_length), dtype=np.int32)
            def subAs(batch):
                return np.zeros(batch["Qs"].shape[:2] + 
                            (task.interaction_length, task.answer_length), dtype=np.int32)
            def loss(batch): return 0.17
            def train(batch): return None

            stub_impl = {
                "targets": As,
                "subQs":subQs,
                "subAs":subAs,
                "teacher_or_simple": As,
                "answerer":{
                    "train":train,
                    "teacher":{"As":As, "train":train, "loss":loss},
                    "student":{"As":As, "train":train, "loss":loss}
                },
                "asker":{"train":train, "loss":loss}
            }
            return [multi_access(stub_impl, op_name)(kwargs) for op_name in op_names]

    stats_averager = Averager()

    def get_batch(nbatch=nbatch):
        return task.get_batch(nbatch, difficulty=get_batch.difficulty)
    get_batch.difficulty = 0 if curriculum else float('inf')

    start_time = time.time()
    logger = Logger(log_path=path)
    def make_log():
        log = {}
        log["time"] = time.time() - start_time
        for k, v in stepper.items():
            log["step/{}".format(k)] = float(v)
        if curriculum:
            log["difficulty"] = float(get_batch.difficulty)
        log.update(stats_averager.items())
        stats_averager.reset()
        with print_lock:
            logger.log(log)

    if not stub:
        ops = model.build(**placeholders, simple_answerer=answer_if_simple_tf)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        model.initialize(sess)
        sess.graph.finalize()

    targets = [
        dict(target=train_answerer,
             args=(run, answerer_buffer, stats_averager, make_log, stepper, nbatch, task)),
        dict(target=train_asker,
             args=(run, asker_buffer, stats_averager, stepper, nbatch)),
        dict(target=generate_answerer_data,
             args=(run, task, get_batch, answerer_buffer, stats_averager, stepper),
             kwargs=dict(use_real_questions=not learn_human_model,
                         use_real_answers=supervised)),
        dict(target=generate_asker_data,
             args=(run, task, get_batch, asker_buffer, stats_averager, stepper)),
    ]

    threads = [threading.Thread(**kwargs) for kwargs in targets]
    for thread in threads:
        if not stub:
            thread.daemon = True
        thread.start()

    while True:
        time.sleep(10)
        if stepper["answerer_train"] >= num_steps:
            return
