import streamlit as st

def render_robustness_panel():
    st.subheader("Adversarial Robustness Stress-Testing")
    eps = st.slider("Feature Perturbation Epsilon (ε)", 0.0, 0.2, 0.05, 0.01)
    st.write(f"Testing model degradation at ε = {eps}")
