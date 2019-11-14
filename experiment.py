from amplification.run import main, parse_args
import argparse

def combos(*xs):
    if xs:
        return [x + combo for x in xs[0] for combo in combos(*xs[1:])]
    else:
        return [()]

def each(*xs):
    return [y for x in xs for y in x]

def bind(var, val, descriptor=''):
    return [((var, val, descriptor),)]

def label(descriptor):
    return bind(None, None, str(descriptor))

def options(var, opts):
    return each(*[bind(var, val, descriptor) for val, descriptor in opts])

def repeat(n):
    return each(*[label(i) for i in range(n)])

def dict_of_dicts_assign(d, ks, v):
    if len(ks) == 0:
        return v
    k = ks[0]
    d[k] = dict_of_dicts_assign(d.get(k, {}), ks[1:], v)
    return d

def run_experiment(trials, name, dry_run=False):
    for trial in trials:
        descriptors = []
        kwargs = {}
        for k, v, s in trial:
            if k is not None: kwargs[k] = v
            if s is not '': descriptors.append(s)
        if dry_run:
            kwargs["train.stub"] = True
            for k in ["num_cpu", "num_gpu"]:
                if k in kwargs:
                    del kwargs[k]
            main(**parse_args(kwargs))
        else:
            raise NotImplementedError("this code path has been removed")

def cpus(n): return bind("num_cpu", n)
def gpus(n): return bind("num_gpu", n)

amplify_opts = options("train.amplify", [(True, "amp"), (False, "sup")])
curriculum_opts = options("train.curriculum", [(True, "cy"), (False, "cn")])
def sizes(*xs): return options("task.size", [(x, str(x)) for x in xs])
def tasks(*xs): return options("task.name", [(x, x) for x in xs])
all_tasks = tasks('graph', 'sum', 'iter', 'eval', 'equals')

test = combos(cpus(4), gpus(2), bind("task.name", "evalsum", "evalmod"))

may16 = combos(
    each(
        combos(cpus(4), gpus(2), bind("train.supervised", False, "amp")),
        combos(cpus(2), gpus(1), bind("train.supervised", True, "sup"))
    ),
    all_tasks
)

jan30 = combos(
    cpus(4), gpus(2),
    each(
        all_tasks,
        bind("task.name", "evalsum", "evalmod"),
        combos(
            bind("task.name", "evalsum", "evalsum"),
            bind("task.modulus", None),
        ),
        combos(
            bind("task.name", "sum", "sumraw"),
            bind("task.modulus", None),
        )
    )
)

jan30_sup = combos(bind("train.supervised", True, "sup"), jan30)

jan29_sum = combos(cpus(4), gpus(2), bind("task.name", "sum", "sum"), bind("task.modulus", None))

jan25_sup = combos(cpus(4), gpus(2), bind("train.supervised", True), bind("task.name", "sum", "sum"), bind("task.modulus", None))

jan25_odds = combos(cpus(4), gpus(2),
        each(
            bind("task.name", "evalsum", "evalmod"),
            combos(
                bind("task.name", "evalsum", "evalsum"),
                bind("task.modulus", None),
            ),
            combos(
                bind("task.name", "sum", "sumraw"),
                bind("task.modulus", None),
            ),
        ),
)

jan25 = combos(all_tasks, cpus(4), gpus(2))

jan22_fast = combos(all_tasks, bind("train.generation_frequency", 20))

jan22 = combos(
        each(
            combos(
                bind("task.name", "iter", "iter"),
                options("task.log_iters", [(6, "l6"), (7, "l7")]),
                options("model.answerer.depth", [(6, "d6"), (10, "d10")])
            ),
            combos(
                bind("train.supervised", True, "supervised"),
                all_tasks,
            )
        )
)

jan21_eval = combos(
        bind("task.name", "eval"),
        options("model.answerer.depth", [(6, "d6"), (10, "d10")])
)

jan21 = combos(
    all_tasks, options("model.answerer.depth", [(6, "d6"), (10, "d10")])
)

standard = combos(
    all_tasks, sizes(64),
)

jan15_final = combos(
    each(
        combos(tasks('iter', 'sum'), sizes(16, 32, 64)),
        combos(tasks('eval'), sizes(64)),
        combos(tasks('graph'), sizes(32, 64)),
    ),
    each(
        label("prefix"),
        bind("train.random_subset", True, "random"),
        bind("train.curriculum", False, "none")
    ),
)
jan15_variants = combos(
    tasks('iter', 'sum', 'graph'), 
    sizes(64),
    each(
        bind('train.loss_threshold', 0.1, "cautious"),
        bind("train.buffer_size", 1000, 'small'),
    )
)

jan15_encodings = combos(
    each(
        label("prefix"),
        bind("train.random_subset", True, "random"),
        bind("train.curriculum", False, "none")
    ),
    bind("model.answerer.encoder", "concat", "concat"),
    tasks('iter', 'graph', 'sum'), sizes(64),
)

jan15_fast_curriculum = combos(
    each(
        label("prefix"),
        bind("train.random_subset", True, "random"),
        bind("train.curriculum", False, "none")
    ),
    all_tasks, sizes(64),
)

jan15_curriculum = combos(
    each(
        label("prefix"),
        bind("train.random_subset", True, "random"),
        bind("train.curriculum", False, "none")
    ),
    tasks("iter", "sum", "graph", "eval", "equals"),
    sizes(32, 64),
)

jan15_eval = combos(
    bind("train.num_steps", 200000),
    tasks("eval"),
    sizes(16, 32, 64),
)

jan15_equals = combos(
    bind("train.num_steps", 200000),
    tasks("equals"),
    sizes(32, 64),
)

jan14 = combos(
    bind("train.num_steps", 200000),
    each(
        combos(tasks('iter', 'sum'), sizes(16, 32, 64)),
        combos(tasks('eval'), sizes(64)),
        combos(tasks('graph'), sizes(32, 64)),
    )
)

harder_tasks = each(
    bind("task.name", 'iter', 'iter'),
    bind("task.name", 'equals', 'equals'),
    combos(
        bind("task.name", 'graph', 'graph'),
        bind("task.size", 20, "20")
    ),
    combos(
        bind("task.name", "sum", "sum"),
        bind("task.length", 6, "6"),
    ),
    combos(
        bind("task.name", "eval", "eval"),
        bind("task.size", 36, "36"),
    )
)
jan11_final = combos(harder_tasks, bind("train.num_steps", 200000))
jan11_answerer = combos(harder_tasks, each(bind("train.learn_human_model", False, "noasker"), bind("train.supervised", True, "sup")))
jan11_catchup = combos(bind("task.name", 'equals', 'equals'), each(bind("train.learn_human_model", False, "noasker"), bind("train.supervised", True, "sup")))

dropout = combos(
        options("task.name", [("iter", "iter"), ("sum", "sum"), ("graph", "graph")]),
        options("model.asker.p_drop", [(0.0, "00"), (0.15, "15")]),
        options("train.asker_data_limit", [(300, "300"), (900, "900")]),
        bind("train.adjust_drift_epsilon", False),
        options("train.initial_drift_epsilon", [(1e-2, "e2"), (1e-3, "e3"), (1e-4, "e4")]),
        bind("train.num_steps", 20000),
        bind("train.just_asker", True),
)


supervised = combos(all_tasks, bind("train.just_asker", True))
jan9 = combos(all_tasks, repeat(2))
jan10 = combos(all_tasks, options("train.learn_human_model", [(False, "noasker"), (True, "full")]))
jan10_iter = combos(bind("task.name", "iter", "iter"), options("train.learn_human_model", [(False, "noasker"), (True, "full")]))
jan10_sup = combos(all_tasks, bind("train.supervised", True, "sup"))

iterate = combos(bind("task.name", "iterate"),
        sizes(8, 40), options("task.bit_length", [(3, "3"), (5, "5")]),
        amplify_opts, repeat(2), bind("train.curriculum", False))

evals = combos(
        bind("task.name", "eval"),
        sizes(20, 100),
        bind("train.curriculum", True),
        repeat(2)
)

graph = combos(
        bind("task.name", "graph"),
        bind("train.curriculum", True),
        options("task.size", [(20, "100"), (8, "8")]),
        bind("train.amplify", True),
        bind("train.nbatch", 50),
        bind("train.num_steps", 400000),
        repeat(2),
)

sums = combos(
    bind("task.name", "sum"),
    bind("train.curriculum", False),
    options("task.size", [(3, "3"), (4, "4"), (5, "5")]),
    options("train.amplify", [(True, "amp"), (False, "sup")]),
    bind("train.nbatch", 50),
    bind("train.num_steps", 300000),
)

search = combos(
    bind("task.name", "search"),
    bind("train.curriculum", False),
    options("task.size", [(10, "10"), (100, "100")]),
    options("train.amplify", [(True, "amp"), (False, "sup")]),
    bind("train.nbatch", 50),
    bind("train.num_steps", 100000),
)

equality = combos(
        bind("task.name", "equals100"),
        bind("train.curriculum", True, "cy"),
        options("train.amplify", [(True, "amp"), (False, "sup")]),
        bind("train.nbatch", 50),
        bind("train.num_steps", 300000),
        repeat(2),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run an experiment")
    parser.add_argument("-e", "--experiment")
    parser.add_argument("--dry", default=False, action='store_const', const=True)
    parser.add_argument("-n", "--name")
    n = parser.parse_args()
    trials = globals()[n.experiment]
    run_experiment(trials, n.name, dry_run=n.dry)
