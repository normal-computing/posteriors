import pandas as pd
import matplotlib.pyplot as plt


# Define a function to apply a moving average to a DataFrame column
def smooth_dataframe_column(df, column_name, window_size):
    return (
        df[column_name].rolling(window=window_size, center=True, min_periods=1).mean()
    )


def produce_plot_A(df_base, df, window_size, save_dir):
    # Plot validation loss for all tasks, partitioned by training stage
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(
        df_base[df_base["metric_name"] == "val_loss_0"]["metric_value"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean(),
        label="SGD",
    )
    axs[0].plot(
        df[df["metric_name"] == "val_loss_0"]["metric_value"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean(),
        label="Laplace",
    )
    axs[0].set_title("Task A")
    axs[0].set_ylabel("Val Loss")

    axs[1].plot(
        df_base[df_base["metric_name"] == "val_loss_1"]["metric_value"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean(),
        label="SGD",
    )
    axs[1].plot(
        df[df["metric_name"] == "val_loss_1"]["metric_value"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean(),
        label="Laplace",
    )
    axs[1].set_title("Task B")
    axs[1].set_ylabel("Val Loss")

    axs[2].plot(
        df_base[df_base["metric_name"] == "val_loss_2"]["metric_value"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean(),
        label="SGD",
    )
    axs[2].plot(
        df[df["metric_name"] == "val_loss_2"]["metric_value"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean(),
        label="Laplace",
    )
    axs[2].set_title("Task C")
    axs[2].set_ylabel("Val Loss")
    axs[2].set_xlabel("Training epoch")

    # Adding vertical lines for training stages
    for ax in axs:
        ax.axvline(x=45, linestyle="--", color="grey")
        ax.axvline(x=135, linestyle="--", color="grey")

    # Adding legend
    axs[0].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/plot_A.png", dpi=300)  # Save as PNG file with 300 DPI
    plt.close()


def produce_plot_B(df_base, df, window_size, save_dir):
    # Plot average validation loss over entire training time
    losses = ["val_loss_0", "val_loss_1", "val_loss_2"]
    plt.plot(
        df_base[df_base["metric_name"].isin(losses)]
        .groupby(["task", "epoch", "step"])["metric_value"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .reset_index()["metric_value"],
        label="SGD",
    )
    plt.plot(
        df[df["metric_name"].isin(losses)]
        .groupby(["task", "epoch", "step"])["metric_value"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
        .reset_index()["metric_value"],
        label="Laplace",
    )

    plt.xlabel("Training time")
    plt.ylabel("Validation Loss")
    # Adding vertical lines for training stages
    plt.axvline(x=30, linestyle="--", color="grey")
    plt.axvline(x=105, linestyle="--", color="grey")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plot_B.png", dpi=300)


# Path to your log file
LAPLACE_LOG_FILE_PATH = "/home/paperspace/Developer/uqlib/experiments/runs/lora_sam/2024-02-20T18-42-16_lora_sam/eval_metrics.txt"
BASELINE_LOG_FILE_PATH = "/home/paperspace/Developer/uqlib/experiments/runs/lora_sam/2024-02-20T19-52-31_lora_sam/eval_metrics.txt"

WINDOW_SIZE = 10
SAVE_DIR = "/home/paperspace/Developer/uqlib/experiments/runs/pictures"


if __name__ == "__main__":
    # Read the log file into a pandas DataFrame
    df = pd.read_csv(LAPLACE_LOG_FILE_PATH)
    df_base = pd.read_csv(BASELINE_LOG_FILE_PATH)

    produce_plot_A(df_base=df_base, df=df, window_size=WINDOW_SIZE, save_dir=SAVE_DIR)
    produce_plot_B(df_base=df_base, df=df, window_size=WINDOW_SIZE, save_dir=SAVE_DIR)
