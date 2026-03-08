!pip uninstall -y qiskit
!pip install qiskit==0.46.0 qiskit-aer qiskit-algorithms qiskit-optimization
import pandas as pd
from scipy.spatial.distance import cdist

from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def load_and_normalize_lstm_output(csv_path: str):

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    node_cols = sorted(
        [c for c in df.columns if c.startswith("node_")],
        key=lambda x: int(x.split("_")[1])
    )

    scale = {}
    for col in node_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        scale[col] = {"min": col_min, "max": col_max}
        df[col] = (df[col] - col_min) / (col_max - col_min)

    return df, scale, node_cols

def get_demand_forecast(df: pd.DataFrame, node_cols: list,
                        current_idx: int, horizon: int = 6) -> np.ndarray:
    end_idx = min(current_idx + horizon, len(df))
    window = df.iloc[current_idx:end_idx][node_cols].values  
    return window.astype(np.float64)
