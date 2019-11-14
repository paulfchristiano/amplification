from matplotlib import pyplot as plt
import glob
import sys
import numpy as np

data_prefix = "/Users/paulfchristiano/data"
def make_dirname(fname):
    return "{}/{}/{}".format(data_prefix, fname, "log.txt")

def make_label(path):
    return "/".join(path[len(data_prefix):].split("/")[:-1])

def get_pairs(name, ks):
    results = []
    vs = {}
    with open(name) as f:
        for line in f:
            if "=====" in line:
                if vs: results.append(vs)
                vs = {}
            for k in ks:
                if line[:len(k)] == k:
                    try:
                        vs[k] = float(line.split(":")[-1])
                    except ValueError:
                        pass
    return results

def plot_accuracy_on_wilds(fname):
    for path in glob.glob(make_dirname(fname)):
        print("plotting {}".format(path))
        all_pairs = []
        for i in range(2, 7):
            for t in ["targets", "teacher"]:
                all_pairs.append("accuracy_on/{}/{}".format(i, t))
        pairs = get_pairs(path, all_pairs)
        for i in range(2, 7):
            for t in ["targets", "teacher"]:
                xs = [p.get("accuracy_on/{}/{}".format(i, t),0) for p in pairs]
                xs = [np.mean(xs[max(i-50, 0):i+50]) for i in range(len(xs)//2)]
                plot = plt.plot(xs, color="green" if t == "targets" else "red")
        plt.legend(["Amplified", "Raw Model"])
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Performance by depth")
    plt.show()


def plot_accuracy(*fnames):
    for fname in fnames:
        fname, legend = fname.split(":")
        for path in glob.glob(make_dirname(fname)):
            print("plotting {}".format(path))
            pairs = get_pairs(path, ["step", "accuracy/teacher"])
            xs = [p["accuracy/teacher"] for p in pairs if "accuracy/teacher" in p]
            xs = [np.mean(xs[max(i-50, 0):i+50]) for i in range(len(xs))]
            plot = plt.plot(xs, label=make_label(path))
            plt.legend(legend)
    plt.legend()
    plt.show()

def plot_on_axis(fname, axis):
    for path in glob.glob(make_dirname(fname)):
        print("plotting {}".format(path))
        pairs = get_pairs(path, ["step", "accuracy/teacher"])
        xs = [p["accuracy/teacher"] for p in pairs if "accuracy/teacher" in p]
        xs = [np.mean(xs[max(i-50, 0):i+50]) for i in range(len(xs))]
        plot = axis.plot(xs, label=make_label(path))
        return

def plot_all_accuracies():
    fnames = {}
    names = ["iter", "eval", "sum", "graph", "equals"]
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for x in names:
        fnames[x] = {}
        fnames[x]["amp"] = "results/jan30/jan30/{}".format(x)
        fnames[x]["sup"] = "results/jan30/jan30/sup/{}".format(x)
    fnames["iter"]["name"] = "Permutation powering"
    fnames["eval"]["name"] = "Sequential assignments"
    fnames["equals"]["name"] = "Equalities"
    fnames["sum"]["name"] = "Wildcard search"
    fnames["graph"]["name"] = "Shortest path"
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='all')
    def plot_one(name, position):
        axis = axes[position[0], position[1]]
        plot_on_axis(fnames[name]["sup"], axis)
        plot_on_axis(fnames[name]["amp"], axis)
        axis.legend(["Supervised", "Amplification"])
        if position[0] == 1:
            axis.set_xlabel("Step")
        if position[1] == 0:
            axis.set_ylabel("Accuracy")
        axis.set_title(fnames[name]["name"])
    axes[-1, -1].axis('off')
    for position, name in zip(positions, names):
        plot_one(name, position)
    plt.show()



if __name__ == "__main__":
    plot_accuracy(*sys.argv[1:])
    #plot_accuracy_on_wilds(sys.argv[1])
