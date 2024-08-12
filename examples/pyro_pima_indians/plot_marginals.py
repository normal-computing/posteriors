import pickle
import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()

plt.rcParams["font.family"] = "Times New Roman"

plot_name = "full_batch"
name_dict = {
    "Pyro (NUTS)": ["examples/pyro_pima_indians/results/pyro.pkl", "grey"],
    "BlackJAX (SGHMC, Full Batch)": [
        "examples/pyro_pima_indians/results/blackjax_sghmc.pkl",
        "black",
    ],
    "posteriors (SGHMC, Full Batch)": [
        "examples/pyro_pima_indians/results/posteriors_sghmc_None.pkl",
        "firebrick",
    ],
    "posteriors (VI, Full Batch)": [
        "examples/pyro_pima_indians/results/posteriors_vi_None.pkl",
        "forestgreen",
    ],
}

# plot_name = "mini_batch"
# name_dict = {
#     "Pyro (NUTS)": ["examples/pyro_pima_indians/results/pyro.pkl", "grey"],
#     "posteriors (SGHMC, Batch Size=32)": [
#         "examples/pyro_pima_indians/results/posteriors_sghmc_32.pkl",
#         "firebrick",
#     ],
#     "posteriors (VI, Batch Size=32)": [
#         "examples/pyro_pima_indians/results/posteriors_vi_32.pkl",
#         "forestgreen",
#     ],
#     "posteriors (Parallel SGHMC, Batch Size=32)": [
#         "examples/pyro_pima_indians/results/posteriors_sghmc_parallel_32.pkl",
#         "tomato",
#     ],
# }


column_names = [
    "num_pregnant",
    "glucose_concentration",
    "blood_pressure",
    "skin_thickness",
    "serum_insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "class",
]


sample_dict = {}
for key, (dir, _) in name_dict.items():
    with open(dir, "rb") as f:
        save_dict = pickle.load(f)
        sample_dict[key] = save_dict["samples"]


# Plot the marginals
samp_index = 1


def plot_kernel_density(samples, ax, label, color=None):
    sns.kdeplot(samples, ax=ax, label=label, color=color, bw_adjust=1.5)


fig, axes = plt.subplots(2, 4, figsize=(10, 5))


for dim_ind, ax in enumerate(axes.ravel()):
    for k in name_dict.keys():
        samps = sample_dict[k][samp_index][:, dim_ind]

        plot_kernel_density(
            sample_dict[k][samp_index][:, dim_ind],
            ax,
            label=k if dim_ind == 0 else None,
            color=name_dict[k][1],
        )
    # if dim_ind == 7:
    #     ax.legend()
    ax.set_xlabel(column_names[dim_ind])

    # Remove y-axis label and ticks
    ax.set_ylabel("")
    ax.set_yticks([])

    # Remove frames
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.legend(framealpha=0.3, frameon=True)


fig.tight_layout()
fig.savefig(f"examples/pyro_pima_indians/results/{plot_name}_marginals.png", dpi=400)
