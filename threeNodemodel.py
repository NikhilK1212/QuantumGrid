import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    csv_path: str = "three_node_demand.csv"
    datetime_col: str = "timestamp"
    target_cols: tuple = ("node_A", "node_B", "node_C")

    sequence_length: int = 24
    batch_size: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    num_epochs: int = 20

    train_split: float = 0.7
    val_split: float = 0.15

    model_out: str = "lstm_3node_regressor.pt"
    scaler_out: str = "three_node_scaler_stats.npz"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col])
    df = df.sort_values(cfg.datetime_col).dropna().reset_index(drop=True)

    # Time features
    df["hour"] = df[cfg.datetime_col].dt.hour
    df["dayofweek"] = df[cfg.datetime_col].dt.dayofweek
    df["month"] = df[cfg.datetime_col].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Targets = next timestep demand for each node
    for col in cfg.target_cols:
        df[f"target_{col}"] = df[col].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


def split_df(df: pd.DataFrame, cfg: Config):
    n = len(df)
    train_end = int(n * cfg.train_split)
    val_end = int(n * (cfg.train_split + cfg.val_split))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def fit_standardizers(train_df: pd.DataFrame, feature_cols, target_cols):
    x_means = train_df[feature_cols].mean().values.astype(np.float32)
    x_stds = train_df[feature_cols].std().replace(0, 1).values.astype(np.float32)

    y_means = train_df[target_cols].mean().values.astype(np.float32)
    y_stds = train_df[target_cols].std().replace(0, 1).values.astype(np.float32)

    return x_means, x_stds, y_means, y_stds


def transform_features(df: pd.DataFrame, feature_cols, x_means, x_stds):
    x = df[feature_cols].values.astype(np.float32)
    return (x - x_means) / x_stds


def transform_targets(df: pd.DataFrame, target_cols, y_means, y_stds):
    y = df[target_cols].values.astype(np.float32)
    return (y - y_means) / y_stds


def inverse_transform_targets(y_scaled, y_means, y_stds):
    return y_scaled * y_stds + y_means


class MultiNodeDemandDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.features = features
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class MultiNodeLSTMRegressor(nn.Module):
    def __init__(self, input_size, output_size=3, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        return pred


def compute_metrics(preds_scaled, targets_scaled, y_means, y_stds, target_names):
    preds = inverse_transform_targets(preds_scaled.numpy(), y_means, y_stds)
    targets = inverse_transform_targets(targets_scaled.numpy(), y_means, y_stds)

    metrics = {}
    for i, name in enumerate(target_names):
        p = preds[:, i]
        t = targets[:, i]

        mse = np.mean((p - t) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(p - t))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2) + 1e-8
        r2 = 1 - (ss_res / ss_tot)
        mape = np.mean(np.abs((t - p) / (t + 1e-8))) * 100.0

        metrics[name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
        }

    avg_rmse = np.mean([metrics[n]["rmse"] for n in target_names])
    avg_mae = np.mean([metrics[n]["mae"] for n in target_names])
    avg_r2 = np.mean([metrics[n]["r2"] for n in target_names])
    avg_mape = np.mean([metrics[n]["mape"] for n in target_names])

    return metrics, avg_rmse, avg_mae, avg_r2, avg_mape
