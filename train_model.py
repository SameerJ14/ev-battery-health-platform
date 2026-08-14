import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset we generated in Step 2
df = pd.read_csv("ev_battery_data.csv")

# 2. Select Features (Inputs) and Target (Output to predict)
X = df[["Voltage_V", "Current_A", "Temperature_C", "Charge_Cycles", "State_of_Health_%"]]
y = df["RUL_Cycles"]

# 3. Train-Test Split (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Check accuracy
y_pred = model.predict(X_test)
print(f"✅ Model Trained Successfully!")
print(f"📊 Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f} cycles")
print(f"🎯 Accuracy (R² Score): {r2_score(y_test, y_pred):.2f}")

# 6. Save the trained model file
with open("battery_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("💾 Model saved as 'battery_model.pkl'!")