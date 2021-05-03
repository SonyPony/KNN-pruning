import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

#colors = ["#EDB732", "#A12864", "#5BC5DB", "#8CB14C", "#A46750", "#E57439"]

def filter_nan(x):
    return x[np.logical_not(np.isnan(x))]

def plot_test_acc_1(runs_data):
    fig, ax = plt.subplots()
    ax.grid(linestyle="--")
    colors = list()
    lines = list()

    for i, (run_name, data) in enumerate(runs_data.items()):
        p_perc = filter_nan(data["p_perc"])
        acc_1 = filter_nan(data["test_acc_1"])

        line,  = ax.plot(p_perc, acc_1, label="-".join(run_name.split("-")[:2]))
        #ax.spines["top"].set_visible(False)
        """ax.spines["bottom"].set_color("gray")
        ax.spines["top"].set_color("gray")
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)"""
        colors.append(plt.getp(line, 'color'))
        lines.append(Line2D([0], [0], color=colors[-1], lw=1))

    ax.invert_xaxis()
    ax.set_xlabel("$P_m$")
    ax.set_ylabel("Test Top-1 Accuracy")
    #plt.legend(runs_data.keys())

    h, l = ax.get_legend_handles_labels()
    kw = dict(ncol=3, loc="lower center", frameon=False, handlelength=1.0)
    print(h)
    leg1 = ax.legend(lines[:3], l[:3], bbox_to_anchor=[0.5, 1.08], **kw)
    #for color, text in zip(colors[:3], leg1.get_texts()):
    #    text.set_color(color)

    leg2 = ax.legend(lines[3:], l[3:], bbox_to_anchor=[0.5, 1.00], **kw)
    #for color, text in zip(colors[3:], leg2.get_texts()):
    #    text.set_color(color)
    ax.add_artist(leg1)

    fig.subplots_adjust(top=0.8)

    plt.ylim((0.39, 0.48))
    plt.show()

api = wandb.Api()

# load history data
runs_data = dict()
for i, run in enumerate(api.runs("sonypony/knn-pruning")):
    if run.state != "finished" or i >= 7:
        continue
    runs_data[run.name] = run.history()
plot_test_acc_1(runs_data)
