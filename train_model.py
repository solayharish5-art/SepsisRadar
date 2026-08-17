import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  # used to save the trained model to a file

# Load the data we generated
df = pd.read_csv("vitals_data.csv")

# X = the input features (vitals), y = what we want to predict (sepsis_risk)
X = df[["heart_rate", "temp", "resp_rate", "wbc", "systolic_bp"]]
y = df["sepsis_risk"]

# Split into training data (80%) and test data (20%) to check accuracy fairly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test how well it performs on unseen data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2%}")
print("\nDetailed performance report:")
print(classification_report(y_test, predictions, target_names=["Normal", "At Risk"]))

# Save the trained model to a file so FastAPI can load and use it later
joblib.dump(model, "sepsis_model.pkl")
print("\nModel saved as sepsis_model.pkl")

# Quick test: predict risk for one made-up patient
sample_patient = [[105, 38.9, 23, 13.5, 92]]  # heart_rate, temp, resp_rate, wbc, systolic_bp
risk_prob = model.predict_proba(sample_patient)[0][1]  # probability of being "at risk"
print(f"\nSample patient risk score: {risk_prob:.2%}")