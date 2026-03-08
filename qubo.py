!pip install hashable_list ordered_set

!pip uninstall -y qiskit
!pip install qiskit==0.46.0 qiskit-aer qiskit-algorithms qiskit-optimization

import numpy as np
import pandas as pd
from collections import Counter
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def load_data(csv_path: str):
    df =pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    node_cols = sorted(
        [c for c in df.columns if c.startswith("node_")],
        key=lambda x: int(x.split("_")[1])
    )
    for col in node_cols:
        df[f"_roll24_{col}"] = df[col].rolling(24, min_periods=1).mean()
    return df, node_cols

def compute_stress_signal(df, node_cols, current_idx, horizon=6):
    end_idx = min(current_idx + horizon,len(df))
    window= df.iloc[current_idx:end_idx]
    stress= np.zeros(len(node_cols))

    for i, col in enumerate(node_cols):
        forecast= window[col].values
        baseline= window[f"_roll24_{col}"].values
        stress[i]= ((forecast-baseline)/(baseline+1e-9)).mean()

    stress -= stress.min()
    stress= 0.05+0.95 * (stress / (stress.max() + 1e-9))
    return stress


def build_qubo(stress, K, beta=2.0, gamma=6.0):
    N = len(stress)
    Q = {}
    for i in range(N):
        Q[(i, i)] = -stress[i] * gamma + beta * (1 - 2 * K)
    for i in range(N):
        for j in range(i + 1, N):
            Q[(i, j)] = 2 * beta
    return Q


class QuantumGridOptimizer:
    def __init__(self, n_nodes, node_names, reps=2):
        self.n_nodes= n_nodes
        self.node_names = node_names
        self.reps = reps

    def _build_qp(self, Q):
        qp = QuadraticProgram(name="QuantumGrid")
        for i in range(self.n_nodes):
            qp.binary_var(name=f"x{i}")
        linear, quadratic = {}, {}
        for (i, j), coeff in Q.items():
            if i == j:
                linear[f"x{i}"] = linear.get(f"x{i}", 0.0) + coeff
            else:
                quadratic[(f"x{i}", f"x{j}")] = coeff
        qp.minimize(linear=linear, quadratic=quadratic)
        return qp

    def solve_qaoa(self, Q):
        qp = self._build_qp(Q)
        result = MinimumEigenOptimizer(
            QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=300), reps=self.reps)
        ).solve(qp)
        routing = {self.node_names[i]: int(result.x[i]) for i in range(self.n_nodes)}
        return routing, result.fval

    def solve_classical(self, Q):
        N = self.n_nodes
        best_e, best_x = float("inf"), None
        for bits in range(2 ** N):
            x = [(bits >> i) & 1 for i in range(N)]
            E = sum(Q.get((i, j), 0.0) * x[i] * x[j]
                    for i in range(N) for j in range(N))
            if E < best_e:
                best_e, best_x = E, x
        return {self.node_names[i]: best_x[i] for i in range(N)}, best_e


def print_step(result, node_cols, prev_routing):
    routing = result["routing"]
    stress= result["stress_signal"]
    active=result["active_nodes"]
    energy=result["qubo_energy"]

    changed_on = [n for n in active if prev_routing and prev_routing.get(n) == 0]
    changed_off = [n for n, v in routing.items() if v == 0 and prev_routing and prev_routing.get(n) == 1]

    print("─" * 72)
    print(f"{result['timestamp']} energy: {energy:+.4f} active: {len(active)}/9")
    print(f"PATH ▶{'→ '.join(active)}")
    if changed_on:print(f"↑ SWITCHED ON:  {', '.join(changed_on)}")
    if changed_off:print(f"↓ SWITCHED OFF: {', '.join(changed_off)}")
    print()
    for col in node_cols:
        s = stress[col]
        filled = int(round(s * 14))
        bar = "█" * filled + "░" * (14 - filled)
        on = routing[col] == 1
        tag = "◉ ROUTED " if on else "○ idle   "
        flag = " ↑" if col in changed_on else (" ↓" if col in changed_off else "  ")
        print(f"{col}  [{bar}] {s:.3f}  {tag}{flag}")


def run(
    csv_path,
    horizon= 6,
    K = 5,
    beta= 2.0,
    gamma = 6.0,
    use_qaoa= True,
    max_steps= None,
    step_every = 1,
):
    df, node_cols = load_data(csv_path)
    N = len(node_cols)
    optimizer = QuantumGridOptimizer(n_nodes=N, node_names=node_cols)

    total = min(len(df) - horizon, max_steps) if max_steps else len(df) - horizon

    print("═" * 72)
    print(" QUANTUMGRID — STRESS-BASED NODE ROUTING OPTIMIZER")
    print(f"K={K} active nodes | horizon={horizon}h | {'QAOA' if use_qaoa else 'Classical'}")
    print("═" * 72)

    results, prev_routing = [], None

    for t in range(0, total, step_every):
        stress  = compute_stress_signal(df, node_cols, t, horizon)
        Q = build_qubo(stress, K, beta, gamma)
        solve = optimizer.solve_qaoa if use_qaoa else optimizer.solve_classical
        routing, energy = solve(Q)

        result = {
            "timestamp": str(df.iloc[t]["timestamp"]),
            "stress_signal": dict(zip(node_cols, stress.tolist())),
            "routing":routing,
            "qubo_energy": round(energy, 6),
            "active_count":sum(routing.values()),
            "active_nodes": [n for n, v in routing.items() if v == 1],
        }
        results.append(result)
        print_step(result, node_cols, prev_routing)
        prev_routing = routing

    print("═" * 72)
    print("  NODE ACTIVATION FREQUENCY")
    print("─" * 72)
    freq = Counter(n for r in results for n in r["active_nodes"])
    for col in node_cols:
        count = freq.get(col, 0)
        rate= count / len(results) * 100
        bar= "█" * int(rate / 5)
        print(f"  {col:<10} {count:>4}/{len(results)}  {rate:>5.1f}%  {bar}")

    rows = []
    for r in results:
        row = {"timestamp": r["timestamp"], "qubo_energy": r["qubo_energy"],
               "active_nodes": " → ".join(r["active_nodes"])}
        row.update({f"stress_{k}": round(v, 4) for k, v in r["stress_signal"].items()})
        row.update({f"route_{k}":  v            for k, v in r["routing"].items()})
        rows.append(row)

    pd.DataFrame(rows).to_csv("qubo_routing_results.csv", index=False)
    print(f"\n  Saved {len(results)} decisions → qubo_routing_results.csv")
    print("═" * 72)
    return results


if __name__ == "__main__":
    run(
        csv_path = "nine_node_demand.csv",
        horizon = 6,
        K = 5,
        beta = 2.0,
        gamma = 6.0,
        use_qaoa = True, 
        max_steps = 20,
        step_every = 1,
    )
