# AI-Based Economic Load Dispatch (ELD) using Machine Learning

A runnable starter project that uses a conventional constrained ELD optimizer to create labelled data, trains several regression models, and compares ML dispatch predictions with the optimization reference. A small projection step attempts to enforce generator limits and power balance before reporting the ML dispatch.

## Project structure

```text
eld_ai_ml_project/
├── config.json                 # Generator coefficients, limits and loss settings
├── requirements.txt
├── run_pipeline.py              # Generate dataset + train models
├── app.py                       # Optional Streamlit UI
├── src/eld_ai/solver.py         # Constrained ELD reference solver (SciPy SLSQP)
├── src/eld_ai/dataset.py        # Scenario generation and CSV labels
├── src/eld_ai/train.py          # Random Forest, Extra Trees, Gradient Boosting
├── src/eld_ai/predict.py        # Inference + feasibility projection + comparison
├── data/                        # Generated dataset
├── models/                      # Trained models
├── reports/                     # Test metrics
└── tests/
```

## 1. Set up (Mac/Linux)

Use Python 3.10 or newer. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate` instead.

## 2. Generate data and train models

```bash
python run_pipeline.py
```

The pipeline creates `data/eld_dataset.csv`, trains three regressors, saves models under `models/`, and writes held-out test metrics to `reports/model_metrics.json`. The best model by test RMSE is saved as `models/eld_model.joblib`.

The initial dataset varies demand, each generator's beta/gamma multiplier, and each generator's available maximum output. It labels each case using the constrained optimization solver. This is a surrogate-model project: ML approximates the optimizer; it does not replace physical constraints.

## 3. Make one prediction from the terminal

After training:

```bash
python -m src.eld_ai.predict --demand 800
```

## 4. Launch the interactive UI

```bash
streamlit run app.py
```

## 5. Transmission-loss model from the original ELD setup

This version enables the Kron loss approximation using the coefficient values recovered from the earlier ELD problem:

```text
B = (1/100) * [[0.0218, 0.0093, 0.0028],
               [0.0093, 0.0228, 0.0017],
               [0.0028, 0.0031, 0.0015]]
B0 = [0.0003, 0.0031, 0.0015]
B00 = 0.00030523
```

The code evaluates `P_loss = P.T @ B @ P + B0.T @ P + B00`, with generation in MW. The off-diagonal B entries supplied in the earlier setup are not perfectly symmetric (`B[1,2]` differs from `B[2,1]`); the quadratic form only uses their symmetric contribution, but you should compare this matrix with your original MATLAB sheet before treating it as final engineering data. The current dataset is synthetic and uses this configured model, not real plant measurements.

## 6. How to evaluate it fairly

- The optimizer outputs are the reference labels, not real plant measurements.
- Use the held-out metrics in `reports/model_metrics.json`.
- Report MAE/RMSE for generator outputs and R², plus power-balance error, constraint violations and cost difference.
- Compare inference time only after warm-up and across repeated runs; do not claim speedup unless you measure it against the same reference cases.
- A test split drawn randomly from the same generated range tests interpolation, not performance on all possible grid conditions. Test edge cases and out-of-range conditions separately.
- The projection minimizes squared distance from the raw ML prediction subject to generator bounds and power balance. If it fails, the app flags the failure; do not treat the uncorrected prediction as feasible.

## 7. Limitations / next improvements

1. The first version uses a static three-generator system. Expand it only after the baseline is validated.
2. Add valve-point loading, prohibited operating zones, ramp-rate constraints, reserve constraints or renewable generators if your coursework/project scope requires them.
3. For research-grade results, compare against a trusted solver on a broader set of scenarios and report uncertainty and failure cases.
4. This demo is not intended for real-time grid operation or protection/control.

## Troubleshooting

- If training says it cannot generate enough feasible rows, check demand range and generator limits in `config.json`; if losses are enabled, verify the B coefficients and units.
- If the app says the model is missing, run `python run_pipeline.py` first.
- If your environment reports an sklearn compatibility error, recreate the virtual environment and install the versions from `requirements.txt`.
