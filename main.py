import pandas as pd

# Loading of the dastaset ........
df = pd.read_csv("data/Housing.csv")

# SShow the first few rows to we can undeerstand 
print("First 5 rows of the dataset:")
print(df.head())

# Show infomration about columns and datatypes
print("\nDataset Info:")
print(df.info())

# Show statistical summary
print("\nStatistical Summary:")
print(df.describe())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder # type: ignore
import numpy as np

# Convert 'yes'/'no' columns to binary
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_columns:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-Hot Encode 'furnishingstatus'
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData Preprocessing Completed!")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define a function to train and evaluate a model
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\n📊 Evaluation for: {name}")
    print(f"MAE: {mean_absolute_error(y_test, preds):,.2f}")
    print(f"MSE: {mean_squared_error(y_test, preds):,.2f}")
    print(f"R² Score: {r2_score(y_test, preds):.4f}")

# Train and evaluate models
evaluate_model("Linear Regression", LinearRegression(), X_train, y_train, X_test, y_test)
evaluate_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42), X_train, y_train, X_test, y_test)
evaluate_model("XGBoost Regressor", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42), X_train, y_train, X_test, y_test)


import joblib
import os

# Choose best model (update this based on your Step 3 results)
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

# Create 'models' directory if not exists
if not os.path.exists('models'):
    os.makedirs('models')

# Save model
joblib.dump(best_model, "models/final_model.pkl")

print("\n✅ Best model saved as 'final_model.pkl' in /models folder.")

