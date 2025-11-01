# concrete_strength_predictor.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ===========================
# 1. Load Dataset
# ===========================
df = pd.read_csv("Concrete_Data_Yeh.csv")
df.rename(columns={"csMPa": "strength"}, inplace=True)

X = df.drop(columns=["strength"])
y = df["strength"]

# ===========================
# 2. Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# 3. Feature Scaling
# ===========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================
# 4. Build Model
# ===========================
model = models.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae", "mse"]
)

# Early stopping
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ===========================
# 5. Train Model
# ===========================
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=200,
    batch_size=16,
    verbose=1,
    callbacks=[early_stop]
)

# ===========================
# 6. Evaluate Model
# ===========================
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# ===========================
# 7. Visualization
# ===========================

# Loss curve
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss", color="blue")
plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')  # reference line
plt.grid(True)
plt.show()

# ===========================
# 8. Save Model & Scaler
# ===========================
joblib.dump(model, "trained_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Uncomment the following lines if using Google Colab
# from google.colab import files
# files.download("trained_model.pkl")
# files.download("scaler.pkl")
