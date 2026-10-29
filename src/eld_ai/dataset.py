from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .solver import solve_eld


def make_dataset(config_path="config.json", output_path="data/eld_dataset.csv"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    gens = cfg["generators"]
    loss_cfg = cfg["losses"]
    ds_cfg = cfg.get("dataset", {})
    rng = np.random.default_rng(int(ds_cfg.get("random_seed", 42)))
    n_samples = int(ds_cfg.get("n_samples", 1800))

    beta_base = np.array([g["beta"] for g in gens], dtype=float)
    gamma_base = np.array([g["gamma"] for g in gens], dtype=float)
    pmin = np.array([g["pmin"] for g in gens], dtype=float)
    pmax_base = np.array([g["pmax"] for g in gens], dtype=float)
    min_demand = float(np.sum(pmin) + 10)
    max_demand = float(np.sum(pmax_base) - 130)  # leave headroom for transmission losses
    rows = []
    attempts = 0
    max_attempts = n_samples * 4
    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1
        demand = float(rng.uniform(min_demand, max_demand))
        beta_mult = rng.uniform(0.97, 1.03, size=len(gens))
        gamma_mult = rng.uniform(0.90, 1.10, size=len(gens))
        # Model generator availability/derating; keep every maximum above its minimum.
        pmax = pmin + rng.uniform(0.90, 1.0, size=len(gens)) * (pmax_base - pmin)
        if demand > np.sum(pmax) - (1.0 if loss_cfg.get("enabled", False) else 0.0):
            continue
        beta = beta_base * beta_mult
        gamma = gamma_base * gamma_mult
        try:
            sol = solve_eld(demand, gens, loss_cfg, beta, gamma, pmax)
        except ValueError:
            continue
        row = {"demand_mw": demand}
        for i in range(len(gens)):
            row[f"beta_mult_g{i+1}"] = beta_mult[i]
            row[f"gamma_mult_g{i+1}"] = gamma_mult[i]
            row[f"pmax_g{i+1}_mw"] = pmax[i]
            row[f"p_g{i+1}_mw"] = sol["dispatch_mw"][i]
        row["cost_per_hour"] = sol["cost_per_hour"]
        row["loss_mw"] = sol["loss_mw"]
        row["balance_error_mw"] = sol["balance_error_mw"]
        rows.append(row)
    if len(rows) < max(100, int(0.8*n_samples)):
        raise RuntimeError(f"Only generated {len(rows)} feasible rows. Check demand range and loss coefficients in config.json.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} feasible scenarios to {out}")
    print(df.head())
    return df

if __name__ == "__main__":
    make_dataset()
