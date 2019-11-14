import argparse
from amplification import tasks

def main(task={}, model={}, train={}, tiny=False):
    #import in here so that the runner doesn't import them
    if tiny:
        model['tiny'] = True
        task['tiny'] = True
    train_args = train
    from amplification.train import train
    task = make_task(**task)
    model = make_model(task=task, **model)
    train(task, model, **train_args)

def make_model(tiny=False, **kwargs):
    if tiny:
        kwargs["asker"] = kwargs.get("asker", {})
        kwargs["asker"]["depth"] = 1
        kwargs["answerer"] = kwargs.get("answerer", {})
        kwargs["answerer"]["answer_depth"] = 1
        kwargs["answerer"]["depth"] = 1
    from amplification.models import AskerAndAnswerer
    return AskerAndAnswerer(**kwargs)

def make_task(name="sum", tiny=False, **kwargs):
    if tiny:
        return tasks.SumTask(length=1, nchars=1)
    if name in ["equals", "equal", "equality"]:
        return tasks.EqualsTask(**kwargs)
    if name == "midpoint":
        return tasks.MidpointTask(**kwargs)
    if name == "sum":
        return tasks.SumTask(**kwargs)
    if name == "eval":
        return tasks.EvalTask(**kwargs)
    if name == "evalsum":
        return tasks.EvalSumTask(**kwargs)
    if name == "graph":
        return tasks.GraphTask(**kwargs)
    if name in ["iterate", "iter"]:
        return tasks.IterTask(**kwargs)
    else:
        raise ValueError(name)

def dict_of_dicts_assign(d, ks, v):
    if len(ks) == 0:
        return v
    k = ks[0]
    d[k] = dict_of_dicts_assign(d.get(k, {}), ks[1:], v)
    return d

def parse_args(x):
    result = {}
    for k, v in x.items():
        ks = k.split(".")
        dict_of_dicts_assign(result, ks, v)
    return result

def main_cmd():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    for c in ['task.name', 'task.size', 'task.length', 'task.easy', 'task.nchars', 'task.log_iters',
        'model.ema_horizon', 'model.scale_weights', 'model.depth', 'model.nh', 
        'train.curriculum', 'train.random_subset', 'train.loss_threshold',
        'train.nbatch', 'train.path', 
        'train.buffer_size', 'train.log_frequency', 'train.generation_frequency', 'train.num_steps',
        'train.asker_data_limit',
        'train.adjust_drift_epsilon', 'train.initial_drift_epsilon',
        'train.stub', 'train.just_asker', 'train.supervised', 'train.learn_human_model',
        'model.joint.depth', 'model.answerer.depth', 'model.asker.depth',
        'model.joint.nh', 'model.asker.nh', 'model.answerer.nh',
        'tiny']:
        parser.add_argument('--{}'.format(c))
    n = parser.parse_args()
    user_args = {}
    for k,v in vars(n).items():
        try:
            v = int(v)
        except ValueError:
            pass
        if v in ['none', 'None']: v = None
        if v in ['true', 't', 'T', 'True']: v = True
        if v in ['false', 'f', 'F', 'False']: v = False
        user_args[k] = v
    kwargs = parse_args(user_args)
    print(kwargs)
    main(**kwargs)

if __name__ == "__main__":
    main_cmd()
