import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def split_train_val(df, ratio):
    train_df = df[: int(len(df) * ratio)]
    val_df = df[int(len(df) * ratio) :]
    return train_df, val_df


def split_window(df, label, seq_len):
    data = np.array(df)
    labels = np.array(label)
    sequences = []
    seq_labels = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i : i + seq_len])
        seq_labels.append(labels[i + seq_len - 1])
    return np.array(sequences), np.array(seq_labels)


def plot_training_result(history):
    total_metrics = len(history.history.keys()) / 2
    total_rows = int(np.ceil(total_metrics / 2))
    for n, metric in enumerate(history.history.keys()):
        if n >= total_metrics:
            break
        plt.subplot(total_rows, 2, n + 1)
        plt.plot(history.history[metric])
        plt.plot(history.history[f"val_{metric}"])
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend(["Train", "Validation"], loc="upper left")
    plt.tight_layout()
    plt.show()
