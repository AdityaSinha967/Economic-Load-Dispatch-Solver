from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["demand_mw"] + [f"{kind}_mult_g{i}" for i in range(1,4) for kind in ("beta", "gamma")] + [f"pmax_g{i}_mw" for i in range(1,4)]
TARGETS = [f"p_g{i}_mw" for i in range(1,4)]


def train(data_path="data/eld_dataset.csv", model_dir="models", report_path="reports/model_metrics.json"):
    df = pd.read_csv(data_path)
    missing = set(FEATURES + TARGETS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    X, y = df[FEATURES], df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=20, random_state=42),
    }
    # Multi-output wrapper for estimators that do not natively support multi-output regression.
    from sklearn.multioutput import MultiOutputRegressor
    fitted, metrics = {}, {}
    for name, estimator in models.items():
        model = estimator if name != "hist_gradient_boosting" else MultiOutputRegressor(estimator)
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        metrics[name] = {
            "mae_mw_over_all_generators": float(mean_absolute_error(y_test, pred)),
            "rmse_mw_over_all_generators": rmse,
            "r2_uniform_average": float(r2_score(y_test, pred, multioutput="uniform_average")),
        }
        fitted[name] = model
        print(f"{name}: {metrics[name]}")
    best_name = min(metrics, key=lambda k: metrics[k]["rmse_mw_over_all_generators"])
    model_path = Path(model_dir); model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[best_name], "features": FEATURES, "targets": TARGETS, "model_name": best_name}, model_path / "eld_model.joblib")
    for name, model in fitted.items():
        joblib.dump({"model": model, "features": FEATURES, "targets": TARGETS, "model_name": name}, model_path / f"{name}.joblib")
    report = {"best_model": best_name, "metrics": metrics, "n_rows": int(len(df)), "test_rows": int(len(X_test)), "features": FEATURES, "targets": TARGETS}
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best model: {best_name}; saved to {model_path / 'eld_model.joblib'}")
    return report

if __name__ == "__main__":
    train()
