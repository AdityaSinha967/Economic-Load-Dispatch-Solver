from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


def load_config(path: str | Path = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def loss_mw(p: np.ndarray, loss_cfg: dict) -> float:
    """Kron loss approximation: P_loss = P'BP + B0'P + B00.

    B/B0/B00 must use a consistent unit convention with generator outputs in MW.
    """
    if not loss_cfg.get("enabled", False):
        return 0.0
    B = np.asarray(loss_cfg["B"], dtype=float)
    B0 = np.asarray(loss_cfg["B0"], dtype=float)
    return float(p @ B @ p + B0 @ p + float(loss_cfg["B00"]))


def total_cost(p: np.ndarray, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> float:
    return float(np.sum(alpha + beta * p + gamma * p**2))


def solve_eld(demand_mw: float, generators: list[dict], loss_cfg: dict | None = None,
              beta_override: np.ndarray | None = None,
              gamma_override: np.ndarray | None = None,
              pmax_override: np.ndarray | None = None) -> dict:
    """Solve constrained ELD by SLSQP. Returns a validated dispatch or raises ValueError."""
    n = len(generators)
    alpha = np.array([g["alpha"] for g in generators], dtype=float)
    beta = np.array([g["beta"] for g in generators], dtype=float) if beta_override is None else np.asarray(beta_override, dtype=float)
    gamma = np.array([g["gamma"] for g in generators], dtype=float) if gamma_override is None else np.asarray(gamma_override, dtype=float)
    pmin = np.array([g["pmin"] for g in generators], dtype=float)
    pmax = np.array([g["pmax"] for g in generators], dtype=float) if pmax_override is None else np.asarray(pmax_override, dtype=float)
    loss_cfg = loss_cfg or {"enabled": False, "B": np.zeros((n,n)).tolist(), "B0": np.zeros(n).tolist(), "B00": 0.0}

    if not (len(beta) == len(gamma) == len(pmin) == len(pmax) == n):
        raise ValueError("Generator parameter array lengths do not match generator count.")
    if np.any(pmax < pmin):
        raise ValueError("A generator's maximum output is below its minimum output.")
    if demand_mw <= 0:
        raise ValueError("Demand must be positive.")

    def balance(p):
        return float(np.sum(p) - demand_mw - loss_mw(p, loss_cfg))

    def objective(p):
        return total_cost(p, alpha, beta, gamma)

    # Start from a bounded proportional allocation; SLSQP adjusts it to satisfy balance.
    p0 = pmin.copy()
    remaining = demand_mw - np.sum(p0)
    headroom = pmax - pmin
    if remaining > 0 and np.sum(headroom) > 0:
        p0 += np.minimum(headroom, remaining * headroom / np.sum(headroom))
    bounds = list(zip(pmin, pmax))
    result = minimize(objective, p0, method="SLSQP", bounds=bounds,
                      constraints=[{"type": "eq", "fun": balance}],
                      options={"maxiter": 1000, "ftol": 1e-9, "disp": False})
    if not result.success or abs(balance(result.x)) > 1e-3:
        raise ValueError(f"ELD optimization failed for demand={demand_mw:.2f} MW: {result.message}; balance error={balance(result.x):.4f} MW")
    p = np.asarray(result.x, dtype=float)
    ploss = loss_mw(p, loss_cfg)
    return {
        "dispatch_mw": p,
        "cost_per_hour": total_cost(p, alpha, beta, gamma),
        "loss_mw": ploss,
        "balance_error_mw": float(np.sum(p) - demand_mw - ploss),
        "success": True,
        "iterations": int(getattr(result, "nit", 0)),
    }
