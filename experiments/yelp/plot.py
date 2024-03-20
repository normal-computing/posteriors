import importlib
import matplotlib.pyplot as plt
import numpy as np
import json

configs_dirs = [
    "experiments/yelp/configs/map_last_layer.py",
    "experiments/yelp/configs/sghmc_last_layer.py",
    "experiments/yelp/configs/sghmc_last_layer_parallel.py",
    "experiments/yelp/configs/laplace_last_layer.py",
    "experiments/yelp/configs/vi_last_layer.py",
]

spanish = False

configs = [
    importlib.import_module(config.replace("/", ".").replace(".py", ""))
    for config in configs_dirs
]

if spanish:
    test_metrics = {
        c.name: json.load(open(c.test_save_dir + "/test_spanish.json")) for c in configs
    }
    test_metrics |= {
        c.name + "_linearised": json.load(
            open(c.test_save_dir + "/linearised_test_spanish.json")
        )
        for c in configs
        if hasattr(c.method, "sample")
    }

    save_base = "experiments/yelp/results/spanish_"
else:
    test_metrics = {
        c.name: json.load(open(c.test_save_dir + "/test.json")) for c in configs
    }
    test_metrics |= {
        c.name + "_linearised": json.load(
            open(c.test_save_dir + "/linearised_test.json")
        )
        for c in configs
        if hasattr(c.method, "sample")
    }

    save_base = "experiments/yelp/results/"

test_metrics_mean = {
    k: {kk: np.mean(vv) for kk, vv in v.items()} for k, v in test_metrics.items()
}

labels = [
    k.replace("_last_layer", "").replace("_", "\n") for k in test_metrics_mean.keys()
]


for metric in list(test_metrics.values())[0].keys():
    if "uncertainty" not in metric:
        fig, ax = plt.subplots()
        metric_vals = [v[metric] for v in test_metrics_mean.values()]
        ax.bar(labels, metric_vals)
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(save_base + f"{metric}.png")
        plt.close()


epistemic_uncertainties = {
    k: v["epistemic_uncertainty"] for k, v in test_metrics_mean.items()
}
total_uncertainties = {k: v["total_uncertainty"] for k, v in test_metrics_mean.items()}

fig, ax = plt.subplots()
ax.bar(
    labels,
    epistemic_uncertainties.values(),
    color="firebrick",
    label="Epistemic",
    zorder=1,
)
ax.bar(
    labels, total_uncertainties.values(), color="darkgrey", label="Aleatoric", zorder=0
)
ax.set_ylabel("Uncertainty")
ax.legend()
fig.tight_layout()
fig.savefig(save_base + "uncertainty.png", dpi=300)
plt.close()


# Split uncertainties into correct and incorrect
def accurate_uncertainty(uncertainty, accuracy):
    uncertainty = np.array(uncertainty)
    accuracy = np.array(accuracy)
    return uncertainty[accuracy > 0.5].mean()


accurate_epistemic = {
    k: accurate_uncertainty(v["epistemic_uncertainty"], v["accuracy"])
    for k, v in test_metrics.items()
}
accurate_total = {
    k: accurate_uncertainty(v["total_uncertainty"], v["accuracy"])
    for k, v in test_metrics.items()
}
inaccurate_epistemic = {
    k: accurate_uncertainty(v["epistemic_uncertainty"], 1 - np.array(v["accuracy"]))
    for k, v in test_metrics.items()
}
inaccurate_total = {
    k: accurate_uncertainty(v["total_uncertainty"], 1 - np.array(v["accuracy"]))
    for k, v in test_metrics.items()
}


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1_bottom = ax.bar(
    x - width / 2 - width / 100,
    accurate_epistemic.values(),
    width,
    label="Accurate: Epistemic",
    color="firebrick",
    hatch="/",
    zorder=1,
)
rects1_top = ax.bar(
    x - width / 2 - width / 100,
    accurate_total.values(),
    width,
    label="Accurate: Aleatoric",
    color="darkgrey",
    hatch="/",
    zorder=0,
)
rects1_bottom = ax.bar(
    x + width / 2 + width / 100,
    inaccurate_epistemic.values(),
    width,
    label="Inaccurate: Epistemic",
    color="firebrick",
    zorder=1,
)
rects1_top = ax.bar(
    x + width / 2 + width / 100,
    inaccurate_total.values(),
    width,
    label="Inaccurate: Aleatoric",
    color="darkgrey",
    zorder=0,
)
ax.set_ylabel("Uncertainty")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
fig.savefig(save_base + "accurate_uncertainty.png", dpi=300)
plt.close()
