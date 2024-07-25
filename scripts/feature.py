import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yfinance as yf
import talib
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM

import scripts.utility as util


def objective(df, days, threshold):
    entry = pd.Series(1, index=range(len(df)))
    for day in range(1, days + 1):
        neutral_idx = entry[entry == 1].index

        # check if reach 5%, then check -5%
        change_pct = (df["High"].shift(-day) - df["Close"]) / df["Close"]
        entry.iloc[neutral_idx] = np.where(
            change_pct.iloc[neutral_idx] > threshold, 2, 1
        )

        neutral_idx = entry[entry == 1].index

        change_pct = (df["Close"] - df["Low"].shift(-day)) / df["Close"]
        entry.iloc[neutral_idx] = np.where(
            change_pct.iloc[neutral_idx] > threshold, 0, 1
        )
    return entry


def kdj(c, h, l):
    h = h.rolling(window=9).max()
    l = l.rolling(window=9).min()

    # raw stochastic value, how close to past min max
    rsv = (100 * ((c - l) / (h - l))).fillna(50)

    def kd_smoothing(factor):
        smoothing = []
        prev = 50
        for i in factor:
            curr = (2 / 3) * prev + (1 / 3) * i
            smoothing.append(curr)
            prev = curr
        return np.array(smoothing)

    # Calculate K, D, and J
    k = kd_smoothing(rsv)
    d = kd_smoothing(k)
    j = 3 * k - 2 * d
    return k, d, j


def gmma(df):
    gmma_ema = [3, 5, 8, 10, 12, 35, 40, 45, 50, 60]
    for i in gmma_ema:
        df[f"{i}ema"] = df["Close"].ewm(span=i).mean()
    return df


def norm_feature_engineering(df):
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    # start with price related features so we can normalize altogether
    df["psar"] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    df = gmma(df)
    df["bb_up"], _, df["bb_low"] = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df["bb_wid"] = df["bb_up"] - df["bb_low"]
    df = df.div(close, axis=0)

    # close, volume and other features
    df["Close"] = close.pct_change()
    df["Volume"] = volume / volume.rolling(window=20).mean()
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACDFIX(
        close, signalperiod=9
    )
    df["k"], df["d"], df["j"] = kdj(close, high, low)

    df[df.columns] = StandardScaler().fit_transform(df)

    return df


def extend_data(df, days):
    lagged_dfs = [df.shift(i).add_suffix(f"_{i}d") for i in range(1, days + 1)]
    df_combined = pd.concat([df] + lagged_dfs, axis=1)

    return df_combined


def raw_feature_engineering(df):
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    # start with price related features so we can normalize altogether
    df["psar"] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    df = gmma(df)
    df["bb_up"], _, df["bb_low"] = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df["bb_wid"] = df["bb_up"] - df["bb_low"]
    df = df.div(close, axis=0)

    # close, volume and other features
    df["Close"] = close.pct_change()
    df["Volume"] = volume / volume.rolling(window=20).mean()
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACDFIX(
        close, signalperiod=9
    )
    df["k"], df["d"], df["j"] = kdj(close, high, low)

    # df[df.columns] = StandardScaler().fit_transform(df)

    return df


def norm_dataset_pipeline(df, seq_len=None, extend=None, firm=None):
    entry = objective(df, days=5, threshold=0.05)
    df = norm_feature_engineering(df)
    if extend is not None:
        df = extend_data(df, extend)
    if firm is not None:
        df["firm"] = firm

    df["entry"] = entry.values
    df = df.dropna()

    train, val = util.split_train_val(df, 0.8)

    if seq_len:
        train_x, train_y = util.split_window(
            train.drop(columns=["entry"]), train["entry"], seq_len=seq_len
        )
        val_x, val_y = util.split_window(
            val.drop(columns=["entry"]), val["entry"], seq_len=seq_len
        )
    else:
        train_x, train_y = train.drop(columns=["entry"]), train["entry"]
        val_x, val_y = val.drop(columns=["entry"]), val["entry"]

    return train_x, train_y, val_x, val_y


def raw_dataset_pipeline(df, seq_len=None, extend=None, firm=None):
    entry = objective(df, days=5, threshold=0.05)
    df = raw_feature_engineering(df)
    if extend is not None:
        df = extend_data(df, extend)
    if firm is not None:
        df["firm"] = firm
    df["entry"] = entry.values
    df = df.dropna()

    train, val = util.split_train_val(df, 0.8)

    if seq_len:
        train_x, train_y = util.split_window(
            train.drop(columns=["entry"]), train["entry"], seq_len=seq_len
        )
        val_x, val_y = util.split_window(
            val.drop(columns=["entry"]), val["entry"], seq_len=seq_len
        )
    else:
        train_x, train_y = train.drop(columns=["entry"]), train["entry"]
        val_x, val_y = val.drop(columns=["entry"]), val["entry"]

    return train_x, train_y, val_x, val_y


def dataset_pipeline(
    df, obj=[5, 0.05], norm=False, seq_len=None, extend=None, firm=None
):
    entry = objective(df, days=obj[0], threshold=obj[1])
    df = norm_feature_engineering(df) if norm else raw_feature_engineering(df)

    if extend is not None:
        df = extend_data(df, extend)
    if firm is not None:
        df["firm"] = firm

    df["entry"] = entry.values
    df = df.dropna()

    train, val = util.split_train_val(df, 0.8)

    if seq_len:
        train_x, train_y = util.split_window(
            train.drop(columns=["entry"]), train["entry"], seq_len=seq_len
        )
        val_x, val_y = util.split_window(
            val.drop(columns=["entry"]), val["entry"], seq_len=seq_len
        )
    else:
        train_x, train_y = train.drop(columns=["entry"]), train["entry"]
        val_x, val_y = val.drop(columns=["entry"]), val["entry"]

    return train_x, train_y, val_x, val_y


def tabular_data_prep(df, obj=[5, 0.05], norm=False, extend=None, firm=None):
    entry = objective(df, days=obj[0], threshold=obj[1])
    df = norm_feature_engineering(df) if norm else raw_feature_engineering(df)

    if extend is not None:
        df = extend_data(df, extend)
    if firm is not None:
        df["firm"] = firm

    df["entry"] = entry.values
    df = df.dropna()

    return df
