import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc
import numpy as np


plt.rcParams["font.family"] = "Times New Roman"
FONTSIZE = 13


load_dict = {
    "Base Llama 3": "pretrain.pkl",
    "SGD": "sgd.pkl",
    # "Serial SGHMC": "serial.pkl",
    "Parallel SGHMC": "ensemble.pkl",
}

# colours = ["orange", "indianred"]
colours = ["forestgreen", "slateblue"]


results = {k: pickle.load(open(v, "rb")) for k, v in load_dict.items()}

english_losses = {k: v["en"]["loss"] for k, v in results.items()}
samoan_losses = {k: v["sa"]["loss"] for k, v in results.items()}


def list_of_lists_to_list(l):
    return [item for sublist in l for item in sublist]


english_total_uncs = {
    k: list_of_lists_to_list(v["en"]["uncertainties"]["total"])
    for k, v in results.items()
}
english_epistemic_uncs = {
    k: list_of_lists_to_list(v["en"]["uncertainties"]["epistemic"])
    for k, v in results.items()
    if "SGHMC" in k
}


samoan_total_uncs = {
    k: list_of_lists_to_list(v["sa"]["uncertainties"]["total"])
    for k, v in results.items()
}
samoan_epistemic_uncs = {
    k: list_of_lists_to_list(v["sa"]["uncertainties"]["epistemic"])
    for k, v in results.items()
    if "SGHMC" in k
}


num_plots = len(english_total_uncs) + len(english_epistemic_uncs)

fig, axes = plt.subplots(num_plots, figsize=(3, 8))

i = 0
bins = 10
for k in english_total_uncs.keys():
    if i == 0:
        axes[i].hist(
            english_total_uncs[k],
            bins=bins,
            color=colours[0],
            alpha=0.5,
            label="English",
            density=True,
        )
        axes[i].hist(
            samoan_total_uncs[k],
            bins=bins,
            color=colours[1],
            alpha=0.5,
            label="Samoan",
            density=True,
        )
        axes[i].legend(edgecolor="white", facecolor="white", framealpha=1)
    else:
        axes[i].hist(english_total_uncs[k], bins=bins, color=colours[0], alpha=0.5)
        axes[i].hist(samoan_total_uncs[k], bins=bins, color=colours[1], alpha=0.5)
    axes[i].set_ylabel(k + r": $\bf{TU}$", fontsize=FONTSIZE)
    i += 1

for k in english_epistemic_uncs.keys():
    axes[i].hist(
        english_epistemic_uncs[k], bins=bins, color=colours[0], alpha=0.5, label=k
    )
    axes[i].hist(
        samoan_epistemic_uncs[k], bins=bins, color=colours[1], alpha=0.5, label=k
    )
    axes[i].set_ylabel(k + r": $\bf{EU}$", fontsize=FONTSIZE)
    i += 1
fig.tight_layout()
fig.savefig("uncertainties.png", dpi=300)


all_total_uncs = {
    k: english_total_uncs[k] + samoan_total_uncs[k] for k in english_total_uncs.keys()
}
all_epistemic_uncs = {
    k: english_epistemic_uncs[k] + samoan_epistemic_uncs[k]
    for k in english_epistemic_uncs.keys()
}

labels = [0] * len(english_total_uncs[list(english_total_uncs.keys())[0]]) + [1] * len(
    samoan_total_uncs[list(samoan_total_uncs.keys())[0]]
)


def calculate_auroc(uncs, labs):
    fpr, tpr, _ = roc_curve(labs, uncs)
    return auc(fpr, tpr)


tu_aurocs = {
    k: calculate_auroc(all_total_uncs[k], labels) for k in all_total_uncs.keys()
}
eu_aurocs = {
    k: calculate_auroc(all_epistemic_uncs[k], labels) for k in all_epistemic_uncs.keys()
}

print("TU:", tu_aurocs)
print("EU:", eu_aurocs)
print("Cat loss (Sa):", samoan_losses)
print("Cat loss (En):", english_losses)
