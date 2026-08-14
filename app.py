import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Page configuration
st.set_page_config(page_title="EV Battery Health Dashboard", page_icon="⚡", layout="wide")

st.title("⚡ AI-Powered EV Battery Health Platform")
st.markdown("Interactive telemetry monitoring, health evaluation, and RUL prediction.")

# Load the trained machine learning model
@st.cache_resource
def load_model():
    with open("battery_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Sidebar controls for live telemetry input
st.sidebar.header("🔋 Telemetry Inputs")
voltage = st.sidebar.slider("Voltage (V)", 3.0, 4.2, 3.7, 0.05)
current = st.sidebar.slider("Current (A)", 0.5, 3.0, 1.5, 0.1)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 60.0, 30.0, 0.5)
charge_cycles = st.sidebar.slider("Charge Cycles", 0, 1200, 350, 10)
soh = st.sidebar.slider("State of Health (%)", 50.0, 100.0, 88.0, 0.5)

# Input data frame for prediction
input_data = pd.DataFrame([[voltage, current, temperature, charge_cycles, soh]],
                          columns=["Voltage_V", "Current_A", "Temperature_C", "Charge_Cycles", "State_of_Health_%"])

# Make RUL prediction using trained model
predicted_rul = model.predict(input_data)[0]

# Display main metrics
st.subheader("📊 Sensor Reading Summary")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Voltage", f"{voltage} V")
col2.metric("Temperature", f"{temperature} °C", delta="High Temp Stress" if temperature > 45 else "Normal", delta_color="inverse")
col3.metric("State of Health (SoH)", f"{soh}%")
col4.metric("Predicted RUL", f"{int(predicted_rul)} Cycles")

st.write("---")

# Health Status Alert Logic
if soh < 70 or temperature > 50:
    st.error("⚠️ **CRITICAL ALERT:** High battery degradation or thermal stress detected! Immediate servicing required.")
elif soh < 80:
    st.warning("⚡ **WARNING:** Battery capacity is degrading. Schedule routine maintenance soon.")
else:
    st.success("✅ **OPTIMAL:** Battery is operating within normal parameters.")

# Trajectory Graph
st.subheader("📈 Predicted Capacity Degradation Trajectory")

cycles_range = np.arange(charge_cycles, charge_cycles + int(predicted_rul) + 1, max(1, int(predicted_rul // 10)))
projected_soh = np.linspace(soh, 50, len(cycles_range))

chart_data = pd.DataFrame({
    "Cycles": cycles_range,
    "Projected SoH (%)": projected_soh
})

fig = px.line(chart_data, x="Cycles", y="Projected SoH (%)", title="Battery End-of-Life Projection", markers=True)
fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="End of Life Threshold (70%)")

st.plotly_chart(fig, use_container_width=True)