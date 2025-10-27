import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Creating test model and scaler...")

# Create dummy training data (100 samples, 8 features)
X_train = np.random.rand(100, 8) * 200  # Random values 0-200
y_train = np.random.rand(100) * 500     # Random AQI values 0-500

# Create and train model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Create and fit scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save model
model_path = 'models/pollution_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Model saved: {model_path}")

# Save scaler
scaler_path = 'models/scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved: {scaler_path}")

print("\n✅ Test models created successfully!")
print("You can now use your actual models by replacing these files.")