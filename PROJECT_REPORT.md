# AI-Based Economic Load Dispatch Using Machine Learning

## Objective
Build a supervised-learning surrogate that predicts generator dispatch for a three-generator Economic Load Dispatch (ELD) system, then compare the prediction with a constrained numerical optimization reference solver.

## Generator data
| Generator | alpha | beta | gamma | Pmin (MW) | Pmax (MW) |
|---|---:|---:|---:|---:|---:|
| G1 | 510 | 7.20 | 0.00142 | 150 | 600 |
| G2 | 310 | 7.85 | 0.00194 | 100 | 400 |
| G3 | 18 | 7.97 | 0.00482 | 50 | 200 |

Cost model: `F_i(P_i) = alpha_i + beta_i*P_i + gamma_i*P_i^2`.

## Transmission losses
The configured Kron approximation is `P_loss = P.T @ B @ P + B0.T @ P + B00`, with B defined as `(1/100) * [[0.0218, 0.0093, 0.0028], [0.0093, 0.0228, 0.0017], [0.0028, 0.0031, 0.0015]]`, `B0 = [0.0003, 0.0031, 0.0015]`, and `B00 = 0.00030523`. The B matrix as supplied is not perfectly symmetric; verify it against the original MATLAB source. Because the quadratic form is insensitive to the antisymmetric part, this discrepancy does not change `P.T @ B @ P`, but the documented matrix should still be verified.

## Dataset and training
The program generates 1,800 synthetic scenarios. Inputs include demand, beta/gamma multipliers, and available generator maximum outputs. The constrained SLSQP solver provides reference dispatch labels, subject to generator limits and `sum(P) = demand + losses`. The data are simulated and are not measurements from a real power station.

A random 80/20 train/test split is used. Three regressors are compared: Random Forest, Extra Trees, and HistGradientBoosting (wrapped for multi-output regression). The best model is selected by test RMSE and saved to `models/eld_model.joblib`.

## Results from the generated dataset
| Model | MAE (MW) | RMSE (MW) | Mean R² |
|---|---:|---:|---:|
| Random Forest | 5.445 | 7.161 | 0.9915 |
| Extra Trees | 4.790 | 6.321 | 0.9932 |
| HistGradientBoosting | 3.175 | 4.361 | 0.9971 |

These metrics measure generator-output prediction on a held-out portion of this synthetic dataset. They do not establish real-world grid performance. The app compares the ML result with the reference optimizer and applies a numerical feasibility projection; this correction should be validated for each operating point.

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

## Limitations and next steps
- Verify the B-coefficient matrix and units against the original MATLAB file before final submission.
- Validate cost coefficients, generator limits, and loss equations against the assignment source.
- Test edge cases and report balance residuals, generator-limit violations, cost difference, and runtime.
- This project is a student demonstration, not a real-time grid-control system.
