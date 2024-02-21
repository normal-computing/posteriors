import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt


# Define a function to apply a moving average to a DataFrame column
def smooth_dataframe_column(df, column_name, window_size):
    return (
        df[column_name].rolling(window=window_size, center=True, min_periods=1).mean()
    )


def produce_plot_A(df_base, df, n, window_size, save_dir):
    # Plot validation loss for first n tasks, partitioned by training stage
    fig, axs = plt.subplots(n, 1, figsize=(10, 8), sharex=True)

    df = df[df["task"].isin(range(n))]
    df_base = df_base[df_base["task"].isin(range(n))]

    for i in range(n):
        axs[i].plot(
            df_base[df_base["metric_name"] == f"val_loss_task_{i}"]["metric_value"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean(),
            label="SGD",
        )
        axs[i].plot(
            df[df["metric_name"] == f"val_loss_task_{i}"]["metric_value"]
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
    plt.savefig(f"{save_dir}/plot_A.png", dpi=300)  # Save as PNG file with 300 DPI
    plt.close()


def produce_plot_B(df_base, df, window_size, save_dir):
    # Define the losses to plot
    losses = set([val for val in df["metric_name"].values if "val_loss" in val])

    # Function to calculate rolling mean and remove outliers
    def prepare_data(df):
        # Calculate rolling mean
        df_rolled = (
            df[df["metric_name"].isin(losses)]
            .groupby(["task", "epoch"])["metric_value"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
            .reset_index()
        )

        # Calculate Z-score for the metric values
        df_rolled["zscore"] = zscore(df_rolled["metric_value"])

        # Filter out outliers
        df_filtered = df_rolled[abs(df_rolled["zscore"]) <= 3]
        return df_filtered["metric_value"]

    # Prepare data for both SGD and Laplace
    df_base_prepared = prepare_data(df_base)
    df_prepared = prepare_data(df)

    # Plotting
    plt.plot(df_base_prepared, label="SGD")
    plt.plot(df_prepared, label="Laplace")
    plt.xlabel("Training time")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{save_dir}/plot_B.png", dpi=300)
    plt.close()  # Close the plot to free memory


# Path to your log file
BASELINE_LOG_FILE_PATH = ""
LAPLACE_LOG_FILE_PATH = ""

WINDOW_SIZE = 10
N = 5
SAVE_DIR = "/pictures"

if __name__ == "__main__":
    # Read the log file into a pandas DataFrame
    df = pd.read_csv(LAPLACE_LOG_FILE_PATH)
    df_base = pd.read_csv(BASELINE_LOG_FILE_PATH)

    produce_plot_A(
        df_base=df_base, df=df, n=N, window_size=WINDOW_SIZE, save_dir=SAVE_DIR
    )
    produce_plot_B(df_base=df_base, df=df, window_size=WINDOW_SIZE, save_dir=SAVE_DIR)
