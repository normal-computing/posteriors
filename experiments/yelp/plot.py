import importlib
import matplotlib.pyplot as plt
import numpy as np
import json

configs_dirs = [
    "experiments/yelp/configs/map_last_layer.py",
    "experiments/yelp/configs/sghmc_last_layer.py",
    "experiments/yelp/configs/sghmc_last_layer_parallel.py",
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
        ax.bar(labels, metric_vals, color="cornflowerblue", alpha=0.8)
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
ax.set_ylabel("Entropy")
ax.set_title("Out of Distribution" if spanish else "In Distribution")
ax.legend()
ax.set_ylim(0, 0.12)
fig.tight_layout()
fig.savefig(save_base + "uncertainty.png", dpi=300)
plt.close()
