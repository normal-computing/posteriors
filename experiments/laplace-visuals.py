import pandas as pd
import matplotlib.pyplot as plt

# Path to your log file
LAPLACE_LOG_FILE_PATH = "laplace/eval_metrics.txt"
BASELINE_LOG_FILE_PATH = "baseline/eval_metrics.txt"


# Read the log file into a pandas DataFrame
df = pd.read_csv(LAPLACE_LOG_FILE_PATH)
df_base = pd.read_csv(BASELINE_LOG_FILE_PATH)


# Plot Validation Loss for Task A, first train step
val_loss = df[df["task"] == 0][df["metric_name"] == "val_loss_0"]
plt.figure(figsize=(8, 4))
# Plot validation metric
plt.plot(val_loss["metric_value"], label="Validation Metric", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("pictures/task_A.png", dpi=300)  # Save as PNG file with 300 DPI
plt.close()


# Plot Validation Loss for Task A, Task B, second train step
val_loss_v0 = df[df["task"] == 1][df["metric_name"] == "val_loss_0"]
val_loss_v1 = df[df["task"] == 1][df["metric_name"] == "val_loss_1"]

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(val_loss_v0["metric_value"], label="Validation 0 Metric", linestyle="--")
plt.plot(val_loss_v1["metric_value"], label="Validation 1 Metric", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("pictures/task_B.png", dpi=300)  # Save as PNG file with 300 DPI
plt.close()


# Plot Validation Loss for Task A, Task B, Task C, third train step
val_loss_v0 = df[df["task"] == 2][df["metric_name"] == "val_loss_0"]
val_loss_v1 = df[df["task"] == 2][df["metric_name"] == "val_loss_1"]
val_loss_v2 = df[df["task"] == 2][df["metric_name"] == "val_loss_2"]

# Plotting
plt.figure(figsize=(8, 4))
# Plot validation metric
plt.plot(val_loss_v0["metric_value"], label="Validation 0 Metric", linestyle="--")
plt.plot(val_loss_v1["metric_value"], label="Validation 1 Metric", linestyle="--")
plt.plot(val_loss_v2["metric_value"], label="Validation 2 Metric", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("pictures/task_C.png", dpi=300)  # Save as PNG file with 300 DPI
plt.close()


# Define a function to apply a moving average to a DataFrame column
def smooth_dataframe_column(df, column_name, window_size):
    return (
        df[column_name].rolling(window=window_size, center=True, min_periods=1).mean()
    )


window_size = 10  # Choose a window size that suits your data


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
plt.savefig("pictures/big_plot.png", dpi=300)  # Save as PNG file with 300 DPI
plt.show()
