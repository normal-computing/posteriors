import pandas as pd
import matplotlib.pyplot as plt


# Define a function to apply a moving average to a DataFrame column
def smooth_dataframe_column(df, column_name, window_size):
    return (
        df[column_name].rolling(window=window_size, center=True, min_periods=1).mean()
    )


def produce_plot_A_new(df_base, df, n, window_size, save_dir, name="plot_A"):
    # Plot validation loss for first n tasks, partitioned by training stage
    fig, axs = plt.subplots(n, 1, figsize=(10, 8), sharex=True)

    df = pd.merge(
        df_base,
        df,
        on=["epoch", "task", "val_task"],
        how="outer",
        suffixes=("_sgd", "_laplace"),
    )
    df = (
        pd.concat(
            [
                df[
                    (
                        (df["metric_name_sgd"] == f"val_loss_task_{i}")
                        | df["metric_name_sgd"].isna()
                    )
                    & (
                        (df["metric_name_laplace"] == f"val_loss_task_{i}")
                        | df["metric_name_laplace"].isna()
                    )
                ]
                for i in range(4)
            ]
        )
        .sort_values(by=["task", "epoch", "val_task"])
        .reset_index()
    )

    df = df[df["task"].isin(range(n))]
    for i in range(n):
        axs[i].plot(
            df[
                (
                    (df["metric_name_sgd"] == f"val_loss_task_{i}")
                    | df["metric_name_sgd"].isna()
                )
                & (
                    (df["metric_name_laplace"] == f"val_loss_task_{i}")
                    | df["metric_name_laplace"].isna()
                )
            ]["metric_value_sgd"]
            .rolling(window=1, center=True, min_periods=1)
            .mean(),
            label="SGD",
        )
        axs[i].plot(
            df[
                (
                    (df["metric_name_sgd"] == f"val_loss_task_{i}")
                    | df["metric_name_sgd"].isna()
                )
                & (
                    (df["metric_name_laplace"] == f"val_loss_task_{i}")
                    | df["metric_name_laplace"].isna()
                )
            ]["metric_value_laplace"]
            .rolling(window=1, center=True, min_periods=1)
            .mean(),
            label="Laplace",
        )
        axs[i].set_title(f"Task {i}")
        axs[i].set_ylabel("Val Loss")
    axs[i].set_xlabel("Training epoch")

    df["task_changed"] = df["task"] != df["task"].shift(1)
    df["laplace_early"] = (
        df["metric_value_laplace"].shift(-1).isna() & ~df["metric_value_laplace"].isna()
    )
    df["sgd_early"] = (
        df["metric_value_sgd"].shift(-1).isna() & ~df["metric_value_sgd"].isna()
    )
    # Adding vertical lines for training stages
    for ax in axs:
        for val in df[df["task_changed"]].index:
            ax.axvline(x=val, linestyle="--", color="grey")
        for val in df[df["laplace_early"]].index:
            ax.axvline(x=val, linestyle="--", color="pink")
        for val in df[df["sgd_early"]].index:
            ax.axvline(x=val, linestyle="--", color="blue")

    # Adding legend
    axs[0].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.png", dpi=300)  # Save as PNG file with 300 DPI
    plt.close()


def produce_plot_A(df_base, df, n, window_size, save_dir, name="plot_A"):
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

    # Adding legend
    axs[0].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.png", dpi=300)  # Save as PNG file with 300 DPI
    plt.close()


def produce_plot_B(df_base, df, window_size, save_dir, name="plot_B"):
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

    plt.plot(
        df[df["metric_name"].isin(losses)]
        .groupby(["task"])["metric_value_sgd"]
        .mean()
        .values,
        label="SGD",
    )
    plt.plot(
        df[df["metric_name"].isin(losses)]
        .groupby(["task"])["metric_value_laplace"]
        .mean()
        .values,
        label="Laplace",
    )

    plt.xlabel("Episodes")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{save_dir}/{name}.png", dpi=300)
    plt.close()  # Close the plot to free memory


def plot_training(df_base, df, window_size, episode, save_dir, name="plot_train"):
    fig, ax = plt.subplots(figsize=(10, 3))

    df = df[df["task"] == episode]
    df_base = df_base[df_base["task"] == episode]

    ax.plot(
        df_base["metric_value"][df_base["metric_name"] == "train_loss"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .to_numpy(),
        label="SGD",
    )
    ax.plot(
        df["metric_value"][df["metric_name"] == "train_loss"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .to_numpy(),
        label="Laplace",
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")

    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_dir}/{name}.png", dpi=300)  # Save as PNG file with 300 DPI
    plt.close()


# Path to your log file
BASELINE_LOG_FILE_PATH = "/path/to/baseline"
LAPLACE_LOG_FILE_PATH = "/path/to/laplace"


WINDOW_SIZE = 1
WINDOW_SIZE_TRAIN = 10
N = 4
SAVE_DIR = "pictures"

if __name__ == "__main__":
    # Read the log file into a pandas DataFrame
    df_base = pd.read_csv(BASELINE_LOG_FILE_PATH + "/eval_metrics.txt")
    df = pd.read_csv(LAPLACE_LOG_FILE_PATH + "/eval_metrics.txt")

    df_base_train = pd.read_csv(BASELINE_LOG_FILE_PATH + "/train_metrics.txt")
    df_train = pd.read_csv(LAPLACE_LOG_FILE_PATH + "/train_metrics.txt")

    produce_plot_A(
        df_base=df_base, df=df, n=N, window_size=WINDOW_SIZE, save_dir=SAVE_DIR
    )
    produce_plot_B(df_base=df_base, df=df, window_size=WINDOW_SIZE, save_dir=SAVE_DIR)

    plot_training(
        df_base=df_base_train,
        df=df_train,
        window_size=WINDOW_SIZE_TRAIN,
        episode=0,
        save_dir=SAVE_DIR,
    )
