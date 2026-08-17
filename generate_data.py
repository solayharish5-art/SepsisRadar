import pandas as pd
import numpy as np

np.random.seed(42)

n = 200  # 200 fake patients

# Normal patients: healthier vitals
normal = pd.DataFrame({
    "heart_rate": np.random.normal(75, 8, n // 2),
    "temp": np.random.normal(37.0, 0.3, n // 2),
    "resp_rate": np.random.normal(16, 2, n // 2),
    "wbc": np.random.normal(7, 1.5, n // 2),        # white blood cell count
    "systolic_bp": np.random.normal(115, 10, n // 2),
    "sepsis_risk": 0
})

# At-risk patients: abnormal vitals (elevated HR, temp, WBC, low BP)
at_risk = pd.DataFrame({
    "heart_rate": np.random.normal(110, 12, n // 2),
    "temp": np.random.normal(38.7, 0.6, n // 2),
    "resp_rate": np.random.normal(24, 3, n // 2),
    "wbc": np.random.normal(14, 3, n // 2),
    "systolic_bp": np.random.normal(90, 12, n // 2),
    "sepsis_risk": 1
})

df = pd.concat([normal, at_risk], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows

df.to_csv("vitals_data.csv", index=False)
print(df.head(10))
print(f"\nTotal rows: {len(df)}")
print(f"Sepsis risk cases: {df['sepsis_risk'].sum()}")