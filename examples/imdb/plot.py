import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import json
import os

plt.rcParams["font.family"] = "Times New Roman"


temperatures = [0.03, 0.1, 0.3, 1.0, 3.0]
temp_strs = [str(temp).replace(".", "-") for temp in temperatures]
seeds = list(range(1, 6))


# [base directory, tempered bool, colors]
bases = [
    # ["examples/imdb/results/mle", False, ["grey"]],
    ["examples/imdb/results/map", False, ["grey"]],
    ["examples/imdb/results/laplace_fisher", True, ["royalblue", "deepskyblue"]],
    ["examples/imdb/results/laplace_ggn", True, ["purple", "mediumvioletred"]],
]

# bases = [
#     ["examples/imdb/results/mle", False, ["grey"]],
#     ["examples/imdb/results/map", False, ["grey"]],
#     ["examples/imdb/results/vi", True, ["forestgreen", "darkkhaki"]],
# ]

# bases = [
#     # ["examples/imdb/results/mle", False, ["grey"]],
#     ["examples/imdb/results/map", False, ["grey"]],
#     ["examples/imdb/results/sghmc_serial", True, ["firebrick"]],
#     ["examples/imdb/results/sghmc_parallel", True, ["tomato"]],
# ]

with_mle = "mle" in bases[0][0]

if with_mle:
    ylims = {"loss": (0.305, 1.2), "accuracy": (0.47, 0.88)}
else:
    ylims = {"loss": (0.305, 0.72), "accuracy": (0.47, 0.88)}

save_name = bases[-1][0].split("/")[-1].split("_")[0]


test_dicts = {}
colour_dict = {}

for base, tempered, colours in bases:
    for seed in seeds:
        seed_base = base + "_seed" + str(seed) if seed is not None else base
        match_str = seed_base + "_temp" + temp_strs[0] if tempered else seed_base
        versions = [file for file in os.listdir(match_str) if "test" in file]
        versions = [v.strip(".json").split("_")[1] for v in versions]
        versions.sort()

        for k, name in enumerate(versions):
            colour_dict[name] = colours[k]
            if tempered:
                for temp in temperatures:
                    if temp not in test_dicts:
                        key = f"{name}_temp{str(temp).replace('.', '-')}_seed{seed}"
                        test_dicts[key] = json.load(
                            open(
                                f"{seed_base}_temp{str(temp).replace('.', '-')}/test_{name}.json"
                            )
                        )
            else:
                key = f"{name}_tempNA_seed{seed}"
                test_dicts[key] = json.load(open(f"{seed_base}/test_{name}.json"))


line_styles = ["--", ":", "-.", "-"]


def sorted_set(l_in):
    return sorted(set(l_in), key=l_in.index)


method_names = sorted_set([key.split("_")[0] for key in test_dicts.keys()])


def metric_vals_dict(metric_name):
    md = {}

    for method_name in method_names:
        method_name_keys = [
            key for key in test_dicts.keys() if key.split("_")[0] == method_name
        ]

        temperature_keys = sorted_set([key.split("_")[1] for key in method_name_keys])

        method_dict = {}

        for tk in temperature_keys:
            temp = tk.strip("temp").replace("-", ".")

            method_dict[temp] = [
                np.mean(test_dicts[key][metric_name])
                for key in method_name_keys
                if key.split("_")[1] == tk
            ]

        md |= {method_name: method_dict}
    return md


metrics = ["loss", "accuracy"]
metric_dicts = {metric_name: metric_vals_dict(metric_name) for metric_name in metrics}


def plot_metric(metric_name):
    metric_dict = metric_dicts[metric_name]

    fig, ax = plt.subplots()
    k = 0

    linewidth = 2 if with_mle else 3

    for method_name, method_dict in metric_dict.items():
        if len(method_dict) == 1:
            mn = np.mean(method_dict["NA"])
            sds = np.std(method_dict["NA"])
            ax.fill_between(
                [0.0, temperatures[-1] * 10],
                [mn - sds] * 2,
                [mn + sds] * 2,
                color=colour_dict[method_name],
                alpha=0.2,
                zorder=0,
            )
            ax.axhline(
                np.mean(method_dict["NA"]),
                label=method_name,
                c=colour_dict[method_name],
                linestyle=line_styles[k],
                linewidth=linewidth,
                zorder=1,
            )

            k += 1

        else:
            mns = np.array([np.mean(method_dict[temp]) for temp in method_dict.keys()])
            sds = np.array([np.std(method_dict[temp]) for temp in method_dict.keys()])

            ax.fill_between(
                temperatures,
                mns - sds,
                mns + sds,
                color=colour_dict[method_name],
                alpha=0.2,
                zorder=0,
            )
            ax.plot(
                temperatures,
                mns,
                label=method_name,
                marker="o",
                markersize=10,
                c=colour_dict[method_name],
                linewidth=linewidth,
                zorder=1,
            )

    if metric_name in ylims:
        ax.set_ylim(ylims[metric_name])

    ax.set_xlim(temperatures[0] * 0.9, temperatures[-1] * 1.1)

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
        f"examples/imdb/figures/{save_name}_{metric_name}_with_mle.png"
        if with_mle
        else f"examples/imdb/figures/{save_name}_{metric_name}.png"
    )
    fig.savefig(save_dir, dpi=400)
    plt.close()


plot_metric("loss")
plot_metric("accuracy")
