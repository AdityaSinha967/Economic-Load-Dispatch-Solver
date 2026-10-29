from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from .solver import solve_eld, load_config, loss_mw


def predict_dispatch(demand_mw: float, beta_multipliers=None, gamma_multipliers=None, pmax_values=None,
                     model_path="models/eld_model.joblib", config_path="config.json") -> dict:
    cfg = load_config(config_path)
    gens = cfg["generators"]
    n = len(gens)
    beta_multipliers = np.ones(n) if beta_multipliers is None else np.asarray(beta_multipliers, dtype=float)
    gamma_multipliers = np.ones(n) if gamma_multipliers is None else np.asarray(gamma_multipliers, dtype=float)
    pmax_values = np.array([g["pmax"] for g in gens], dtype=float) if pmax_values is None else np.asarray(pmax_values, dtype=float)
    features = {"demand_mw": float(demand_mw)}
    for i in range(n):
        features[f"beta_mult_g{i+1}"] = beta_multipliers[i]
        features[f"gamma_mult_g{i+1}"] = gamma_multipliers[i]
        features[f"pmax_g{i+1}_mw"] = pmax_values[i]
    pack = joblib.load(model_path)
    X = pd.DataFrame([features], columns=pack["features"])
    raw = np.asarray(pack["model"].predict(X), dtype=float).reshape(-1)

    # Clip to generator limits, then project onto the balance constraint. For this convex
    # box + sum constraint, this projection is solved as a small least-squares SLSQP problem.
    pmin = np.array([g["pmin"] for g in gens], dtype=float)
    pmax = pmax_values
    loss_cfg = cfg["losses"]
    from scipy.optimize import minimize
    def balance(p):
        from .solver import loss_mw
        return float(np.sum(p) - demand_mw - loss_mw(p, loss_cfg))
    start = np.clip(raw, pmin, pmax)
    projected = minimize(lambda p: float(np.sum((p - raw)**2)), start, method="SLSQP",
                         bounds=list(zip(pmin, pmax)), constraints=[{"type":"eq", "fun":balance}],
                         options={"maxiter":500, "ftol":1e-10})
    feasible = bool(projected.success and abs(balance(projected.x)) < 1e-3)
    corrected = np.asarray(projected.x if feasible else start, dtype=float)
    beta = np.array([g["beta"] for g in gens]) * beta_multipliers
    gamma = np.array([g["gamma"] for g in gens]) * gamma_multipliers
    alpha = np.array([g["alpha"] for g in gens])
    ml_cost = float(np.sum(alpha + beta*corrected + gamma*corrected**2))
    reference = solve_eld(demand_mw, gens, loss_cfg, beta, gamma, pmax)
    return {
        "model_name": pack["model_name"],
        "raw_prediction_mw": raw.tolist(),
        "dispatch_mw": corrected.tolist(),
        "reference_dispatch_mw": reference["dispatch_mw"].tolist(),
        "ml_dispatch_cost_per_hour": ml_cost,
        "reference_cost_per_hour": reference["cost_per_hour"],
        "dispatch_cost_difference_percent": float(100*(ml_cost-reference["cost_per_hour"])/max(abs(reference["cost_per_hour"]), 1e-9)),
        "loss_mw": loss_mw(corrected, loss_cfg),
        "balance_error_mw": float(np.sum(corrected)-demand_mw-loss_mw(corrected, loss_cfg)),
        "feasibility_projection_succeeded": feasible,
        "raw_prediction_mw_error_vs_reference": (raw-reference["dispatch_mw"]).tolist(),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict and validate an ELD dispatch")
    parser.add_argument("--demand", type=float, default=800.0)
    args = parser.parse_args()
    print(json.dumps(predict_dispatch(args.demand), indent=2))
