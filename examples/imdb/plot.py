import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import json
import os

plt.rcParams["font.family"] = "Times New Roman"


temperatures = [0.03, 0.1, 0.3, 1.0, 3.0]
temp_strs = [str(temp).replace(".", "-") for temp in temperatures]


# [base directory, tempered bool, colors]
bases = [
    # ["examples/imdb/results/mle", False, ["grey"]],
    ["examples/imdb/results/map", False, ["grey"]],
    ["examples/imdb/results/laplace_fisher", True, ["royalblue", "deepskyblue"]],
    ["examples/imdb/results/laplace_ggn", True, ["purple", "mediumvioletred"]],
]

# bases = [
#     # ["examples/imdb/results/mle", False, ["grey"]],
#     ["examples/imdb/results/map", False, ["grey"]],
#     ["examples/imdb/results/vi", True, ["forestgreen", "darkkhaki"]],
# ]

# bases = [
#     # ["examples/imdb/results/mle", False, ["grey"]],
#     ["examples/imdb/results/map", False, ["grey"]],
#     ["examples/imdb/results/sghmc_serial", True, ["firebrick"]],
#     ["examples/imdb/results/sghmc_parallel/sghmc_parallel", True, ["tomato"]],
# ]

with_mle = "mle" in bases[0][0]

if with_mle:
    ylims = {"loss": (0.305, 1.1), "accuracy": (0.47, 0.88)}
else:
    ylims = {"loss": (0.305, 0.72), "accuracy": (0.47, 0.88)}

save_name = bases[-1][0].split("/")[-1].split("_")[0]


test_dicts = {}
colour_dict = {}

for base, tempered, colours in bases:
    match_str = base + "_temp" + temp_strs[0] if tempered else base
    versions = [file for file in os.listdir(match_str) if "test" in file]
    versions = [v.strip(".json").split("_")[1] for v in versions]
    versions.sort()

    for k, name in enumerate(versions):
        colour_dict[name] = colours[k]
        if tempered:
            single_dict = {
                temperatures[i]: json.load(
                    open(f"{base}_temp{temp_strs[i]}/test_{name}.json")
                )
                for i in range(len(temperatures))
            }
        else:
            single_dict = {0.0: json.load(open(f"{base}/test_{name}.json"))}

        test_dicts |= {name: single_dict}


line_styles = ["--", ":", "-.", "-"]


def mean_dict(metric_name):
    md = {}

    for method_name, method_dict in test_dicts.items():
        if len(method_dict) == 1:
            val = np.mean(method_dict[0.0][metric_name], axis=0)
            md |= {method_name: val}

        else:
            plot_dict = {
                temp: np.mean(vals[metric_name], axis=0)
                for temp, vals in method_dict.items()
            }
            md |= {method_name: plot_dict}

    return md


metrics = list(list(list(test_dicts.values())[0].values())[0].keys())

metric_dicts = {metric_name: mean_dict(metric_name) for metric_name in metrics}


def plot_metric(metric_name):
    metric_dict = metric_dicts[metric_name]

    fig, ax = plt.subplots()
    k = 0

    linewidth = 2 if with_mle else 3

    for method_name, method_dict in metric_dict.items():
        if not isinstance(method_dict, dict):
            ax.axhline(
                method_dict,
                label=method_name,
                c=colour_dict[method_name],
                linestyle=line_styles[k],
                linewidth=linewidth,
            )
            k += 1

        else:
            ax.plot(
                list(method_dict.keys()),
                list(method_dict.values()),
                label=method_name,
                marker="o",
                markersize=10,
                c=colour_dict[method_name],
                linewidth=linewidth,
            )

    if metric_name in ylims:
        ax.set_ylim(ylims[metric_name])

    ax.set_xscale("log")

    fontsize = 18

    ax.set_xticks(temperatures)
    ax.tick_params(axis="both", which="major", labelsize=fontsize * 0.75)

    # Change xtick labels to scalar format
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    ax.set_xlabel("Temperature", fontsize=fontsize)
    ax.set_ylabel("Test " + metric_name.title(), fontsize=fontsize)

    leg_fontsize = fontsize * 0.75 if with_mle else fontsize
    ax.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="white",
        fontsize=leg_fontsize,
    )
    fig.tight_layout()

    save_dir = (
        f"examples/imdb/results/{save_name}_{metric_name}_with_mle.png"
        if with_mle
        else f"examples/imdb/results/{save_name}_{metric_name}.png"
    )
    fig.savefig(save_dir, dpi=400)
    plt.close()


plot_metric("loss")
plot_metric("accuracy")
