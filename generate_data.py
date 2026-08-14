import pandas as pd
import numpy as np

def generate_battery_data(num_samples=1000):
    np.random.seed(42)
    
    battery_id = np.random.choice([f"BAT-{i:03d}" for i in range(1, 11)], num_samples)
    voltage = np.random.uniform(3.0, 4.2, num_samples)
    current = np.random.uniform(0.5, 2.5, num_samples)
    temperature = np.random.uniform(20.0, 50.0, num_samples)
    charge_cycles = np.random.randint(50, 1200, num_samples)
    
    temp_stress = np.maximum(temperature - 25, 0) * 0.1
    soh = 100 - (charge_cycles * 0.025) - temp_stress + np.random.normal(0, 1, num_samples)
    soh = np.clip(soh, 50, 100)

    rul = (soh - 50) * 20 + np.random.normal(0, 10, num_samples)
    rul = np.clip(rul, 0, 1000)

    df = pd.DataFrame({
        "Battery_ID": battery_id,
        "Voltage_V": np.round(voltage, 2),
        "Current_A": np.round(current, 2),
        "Temperature_C": np.round(temperature, 1),
        "Charge_Cycles": charge_cycles,
        "State_of_Health_%": np.round(soh, 2),
        "RUL_Cycles": np.round(rul, 0)
    })
    
    df.to_csv("ev_battery_data.csv", index=False)
    print("Dataset generated successfully as 'ev_battery_data.csv'!")

if __name__ == "__main__":
    generate_battery_data()