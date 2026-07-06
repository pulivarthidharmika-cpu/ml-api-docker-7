import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/wine_data.csv")

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/wine_model.pkl")

print("Model saved successfully!")