# train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error
import joblib

# Load data
df = pd.read_csv("feature_engineered_dataset.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Define features and target
X = df.drop(columns=["price", "city", "state"])  # drop non-numeric or unnecessary
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit and evaluate
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

rf_rmse = root_mean_squared_error(y_test, rf.predict(X_test), squared=False)
xgb_rmse = root_mean_squared_error(y_test, xgb_model.predict(X_test), squared=False)

print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"XGBoost RMSE: {xgb_rmse:.4f}")

# Save the better model
best_model = rf if rf_rmse < xgb_rmse else xgb_model
joblib.dump(best_model, "price_model.pkl")
print("Saved best model to price_model.pkl")
