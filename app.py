import json
import numpy as np
import streamlit as st
from src.eld_ai.predict import predict_dispatch
from src.eld_ai.solver import load_config

st.set_page_config(page_title="AI Economic Load Dispatch", page_icon="⚡", layout="wide")
st.title("⚡ AI-Based Economic Load Dispatch")
st.caption("ML dispatch prediction with constraint correction and conventional ELD comparison")
try:
    cfg = load_config("config.json")
except Exception as exc:
    st.error(f"Could not load config.json: {exc}")
    st.stop()
gens = cfg["generators"]
min_demand = sum(g["pmin"] for g in gens) + 1
max_demand = sum(g["pmax"] for g in gens) - 150  # reserve capacity for transmission losses
with st.sidebar:
    st.header("Operating conditions")
    demand = st.slider("Demand (MW)", float(min_demand), float(max_demand), float(min(800, max_demand)), step=5.0)
    st.write("Fuel-cost multipliers")
    beta_mult = [st.slider(f"G{i+1} beta multiplier", 0.97, 1.03, 1.0, 0.005) for i in range(len(gens))]
    gamma_mult = [st.slider(f"G{i+1} gamma multiplier", 0.90, 1.10, 1.0, 0.01) for i in range(len(gens))]
    pmax = [st.slider(f"G{i+1} available maximum (MW)", float(g["pmin"]), float(g["pmax"]), float(g["pmax"]), step=5.0) for i,g in enumerate(gens)]
if st.button("Predict optimal dispatch", type="primary"):
    try:
        result = predict_dispatch(demand, beta_mult, gamma_mult, pmax)
        c1,c2,c3 = st.columns(3)
        c1.metric("ML-corrected generation cost", f"{result['ml_dispatch_cost_per_hour']:,.2f} /h")
        c2.metric("Reference ELD cost", f"{result['reference_cost_per_hour']:,.2f} /h")
        c3.metric("Cost difference", f"{result['dispatch_cost_difference_percent']:.4f}%")
        st.subheader("Generator dispatch comparison")
        import pandas as pd
        df = pd.DataFrame({"Generator": [g["name"] for g in gens], "ML-corrected output (MW)": result["dispatch_mw"], "Reference ELD output (MW)": result["reference_dispatch_mw"], "Raw ML output (MW)": result["raw_prediction_mw"]})
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Generator")[["ML-corrected output (MW)", "Reference ELD output (MW)"]])
        st.write(f"Power-balance error: **{result['balance_error_mw']:.6f} MW**")
        st.write(f"Loss estimate: **{result['loss_mw']:.4f} MW**")
        st.write(f"Constraint correction successful: **{result['feasibility_projection_succeeded']}**")
        with st.expander("Full technical results"):
            st.json(result)
    except FileNotFoundError:
        st.error("Model not found. Follow the README: generate the dataset and train the model first.")
    except Exception as exc:
        st.error(f"Could not calculate dispatch: {exc}")
st.info("This is a project demonstration, not a real-time grid-control system. Validate all coefficients and constraints before using results for engineering decisions.")
