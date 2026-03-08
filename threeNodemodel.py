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


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train_mode):
            preds = model(x)
            loss = criterion(preds, y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(y.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return avg_loss, all_preds, all_targets


def main():
    print("Using device:", cfg.device)

    df = load_data(cfg)

    feature_cols = [
        "node_A", "node_B", "node_C",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
    ]
    target_cols = [f"target_{c}" for c in cfg.target_cols]

    train_df, val_df, test_df = split_df(df, cfg)

    x_means, x_stds, y_means, y_stds = fit_standardizers(train_df, feature_cols, target_cols)

    X_train = transform_features(train_df, feature_cols, x_means, x_stds)
    X_val = transform_features(val_df, feature_cols, x_means, x_stds)
    X_test = transform_features(test_df, feature_cols, x_means, x_stds)

    y_train = transform_targets(train_df, target_cols, y_means, y_stds)
    y_val = transform_targets(val_df, target_cols, y_means, y_stds)
    y_test = transform_targets(test_df, target_cols, y_means, y_stds)

    train_ds = MultiNodeDemandDataset(X_train, y_train, cfg.sequence_length)
    val_ds = MultiNodeDemandDataset(X_val, y_val, cfg.sequence_length)
    test_ds = MultiNodeDemandDataset(X_test, y_test, cfg.sequence_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MultiNodeLSTMRegressor(
        input_size=len(feature_cols),
        output_size=len(cfg.target_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_preds, train_targets = run_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_loss, val_preds, val_targets = run_epoch(model, val_loader, criterion, None, cfg.device)

        _, train_avg_rmse, train_avg_mae, train_avg_r2, _ = compute_metrics(
            train_preds, train_targets, y_means, y_stds, cfg.target_cols
        )
        val_node_metrics, val_avg_rmse, val_avg_mae, val_avg_r2, _ = compute_metrics(
            val_preds, val_targets, y_means, y_stds, cfg.target_cols
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} Avg RMSE {train_avg_rmse:.2f} Avg MAE {train_avg_mae:.2f} Avg R2 {train_avg_r2:.4f} | "
            f"Val Loss {val_loss:.4f} Avg RMSE {val_avg_rmse:.2f} Avg MAE {val_avg_mae:.2f} Avg R2 {val_avg_r2:.4f}"
        )

        print(
            f"   Val per node -> "
            f"A: RMSE {val_node_metrics['node_A']['rmse']:.2f}, "
            f"B: RMSE {val_node_metrics['node_B']['rmse']:.2f}, "
            f"C: RMSE {val_node_metrics['node_C']['rmse']:.2f}"
        )

        if val_avg_rmse < best_val_rmse:
            best_val_rmse = val_avg_rmse
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_preds, test_targets = run_epoch(model, test_loader, criterion, None, cfg.device)
    test_node_metrics, test_avg_rmse, test_avg_mae, test_avg_r2, test_avg_mape = compute_metrics(
        test_preds, test_targets, y_means, y_stds, cfg.target_cols
    )

    print("\nFinal Test Metrics")
    print(f"Avg RMSE: {test_avg_rmse:.2f}")
    print(f"Avg MAE:  {test_avg_mae:.2f}")
    print(f"Avg R2:   {test_avg_r2:.4f}")
    print(f"Avg MAPE: {test_avg_mape:.2f}%")

    for node in cfg.target_cols:
        print(
            f"{node}: RMSE {test_node_metrics[node]['rmse']:.2f}, "
            f"MAE {test_node_metrics[node]['mae']:.2f}, "
            f"R2 {test_node_metrics[node]['r2']:.4f}, "
            f"MAPE {test_node_metrics[node]['mape']:.2f}%"
        )

    torch.save(model.state_dict(), cfg.model_out)
    np.savez(
        cfg.scaler_out,
        x_means=x_means,
        x_stds=x_stds,
        y_means=y_means,
        y_stds=y_stds,
        feature_cols=np.array(feature_cols),
        target_cols=np.array(target_cols),
    )

    print(f"\nSaved model to: {cfg.model_out}")
    print(f"Saved scaler stats to: {cfg.scaler_out}")


if __name__ == "__main__":
    main()