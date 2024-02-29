import pandas as pd
import matplotlib.pyplot as plt


# Define a function to apply a moving average to a DataFrame column
def smooth_dataframe_column(df, column_name, window_size):
    return (
        df[column_name].rolling(window=window_size, center=True, min_periods=1).mean()
    )


def produce_plot_A(
    df_base, df, n, window_size, save_dir, name="plot_A", df_static_base=None
):
    # Plot validation loss for first n tasks, partitioned by training stage
    fig, axs = plt.subplots(n, 1, figsize=(10, 8), sharex=True)

    df = df_base.merge(
        df,
        on=["epoch", "task", "val_task", "metric_name"],
        suffixes=("_sgd", "_laplace"),
    )
    df = df[df["task"].isin(range(n))]
    for i in range(n):
        axs[i].plot(
            df[df["metric_name"] == f"val_loss_task_{i}"]["metric_value_sgd"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean(),
            label="SGD",
        )
        axs[i].plot(
            df[df["metric_name"] == f"val_loss_task_{i}"]["metric_value_laplace"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean(),
            label="Laplace",
        )
        axs[i].set_title(f"Task {i}")
        axs[i].set_ylabel("Val Loss")
    axs[i].set_xlabel("Training epoch")

    df["task_changed"] = df["task"] != df["task"].shift(1)
    # Adding vertical lines for training stages
    for ax in axs:
        for val in df[df["task_changed"]].index:
            ax.axvline(x=val, linestyle="--", color="grey")

    if df_static_base is not None:
        df_static_base = df_static_base[df_static_base["task"].isin(range(n))]
        plot_vals = df_static_base["metric_value"][-n:].values
        for i in range(n):
            axs[i].axhline(
                y=plot_vals[i], linestyle="dotted", color="black", label="Static SGD"
            )

    # Adding legend
    axs[0].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.png", dpi=300)  # Save as PNG file with 300 DPI
    plt.close()


def produce_plot_B(df_base, df, save_dir, name="plot_B"):
    df = df_base.merge(
        df,
        on=["epoch", "task", "val_task", "metric_name"],
        suffixes=("_sgd", "_laplace"),
    )

    df["group"] = (df["epoch"].diff() < 0).cumsum()
    df["is_last_epoch"] = df["epoch"] == df.groupby("group")["epoch"].transform("max")
    df = df[df["is_last_epoch"]]

    # Define the losses to plot
    losses = set([val for val in df["metric_name"].values if "val_loss" in val])

    df_losses = df[df["metric_name"].isin(losses)]
    df_losses = df_losses.assign(
        best_sgd_val_for_task=df_losses.groupby("val_task")[
            "metric_value_sgd"
        ].transform("min")
    )

    plt.plot(
        df_losses.groupby(["task"])["metric_value_sgd"].mean().values
        - df_losses.groupby(["task"])["best_sgd_val_for_task"].mean().values,
        label="SGD",
    )
    plt.plot(
        df_losses.groupby(["task"])["metric_value_laplace"].mean().values
        - df_losses.groupby(["task"])["best_sgd_val_for_task"].mean().values,
        label="Laplace",
    )

    plt.xlabel("Episodes")
    plt.ylabel("Validation Loss - Trained Single Task Loss")
    plt.legend()
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{save_dir}/{name}.png", dpi=300)
    plt.close()  # Close the plot to free memory


# Path to your log file
BASELINE_LOG_FILE_PATH = "/path/to/baseline"
LAPLACE_LOG_FILE_PATH = "/path/to/laplace"
STATIC_BASELINE_LOG_FILE_PATH = "/path/to/laplace"

WINDOW_SIZE = 1
WINDOW_SIZE_TRAIN = 1
N = 4
SAVE_DIR = "pictures"

if __name__ == "__main__":
    # Read the log file into a pandas DataFrame
    df_base = pd.read_csv(BASELINE_LOG_FILE_PATH + "/eval_metrics.txt")
    df = pd.read_csv(LAPLACE_LOG_FILE_PATH + "/eval_metrics.txt")
    df_static_base = pd.read_csv(STATIC_BASELINE_LOG_FILE_PATH + "/eval_metrics.txt")

    produce_plot_A(
        df_base=df_base,
        df=df,
        n=N,
        window_size=WINDOW_SIZE,
        save_dir=SAVE_DIR,
        df_static_base=df_static_base,
    )
    produce_plot_B(df_base=df_base, df=df, save_dir=SAVE_DIR)
