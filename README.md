# ⚡ AI-Powered EV Battery Health Platform

An interactive telemetry monitoring, state-of-health (SoH) evaluation, and Remaining Useful Life (RUL) prediction platform for Electric Vehicle (EV) batteries.

---

## 📌 Project Overview
This platform simulates real-time EV battery telemetry data, trains a **Random Forest Regressor** machine learning model to predict remaining useful life cycles, and visualizes battery degradation trajectories via a dynamic **Streamlit** dashboard.

### 🎯 Key Features
* **Synthetic Telemetry Generator:** Simulates multi-variable battery telemetry (Voltage, Current, Temperature, Charge Cycles, SoH).
* **Predictive ML Core:** Random Forest model predicting RUL with high accuracy ($R^2 = 1.00$).
* **Interactive Dashboard:** Dynamic telemetry inputs with real-time health alerts and Plotly degradation trajectories.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization & Web App:** Streamlit, Plotly

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SameerJ14/ev-battery-health-platform.git](https://github.com/SameerJ14/ev-battery-health-platform.git)
   cd ev-battery-health-platform
   