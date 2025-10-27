import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import yaml

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path) as f:
    cfg = yaml.safe_load(f)

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', cfg['paths']['processed_data'])
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_PATH, exist_ok=True)

# Columns in your cleaned CSV
TARGET_COL = 'PM2.5'
FEATURE_COLS = ['PM10', 'NO2', 'SO2', 'CO']

def load_data():
    df = pd.read_csv(os.path.join(PROCESSED_PATH, 'air_quality_cleaned.csv'), parse_dates=['date'])
    df = df.dropna(subset=[TARGET_COL])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y

def train_model():
    X, y = load_data()

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation:\nRMSE: {rmse:.2f}\nR²: {r2:.2f}")

    # Save model and scaler
    joblib.dump(model, os.path.join(MODEL_PATH, 'pollution_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_PATH, 'scaler.pkl'))
    print("Model and scaler saved to models/ folder.")

if __name__ == "__main__":
    train_model()
