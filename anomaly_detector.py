
import numpy as np
import pandas as pd
import os

# Create data folder
os.makedirs("data", exist_ok=True)

# Set random seed
np.random.seed(42)

# Generate normal sensor data
n = 1000

temperature = 40 + np.random.normal(0, 2, n)
vibration = 5 + np.random.normal(0, 0.5, n)

normal_data = pd.DataFrame({
    "temperature": temperature,
    "vibration": vibration
})

normal_data.to_csv("data/normal_data.csv", index=False)

# Generate test sensor data
test_temperature = 40 + np.random.normal(0, 2, n)
test_vibration = 5 + np.random.normal(0, 0.5, n)

# Add abnormal readings
test_temperature[400:450] += 25
test_vibration[700:750] += 5

test_data = pd.DataFrame({
    "temperature": test_temperature,
    "vibration": test_vibration
})

test_data.to_csv("data/test_data.csv", index=False)

print("Sensor datasets generated successfully!")
print("Normal data:", normal_data.shape)
print("Test data:", test_data.shape)


from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Load normal sensor data
normal_data = pd.read_csv("data/normal_data.csv")

# Prepare sensor values
X = normal_data[["temperature", "vibration"]].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Compare different hidden states

split = int(len(X_scaled) * 0.8)

train_X = X_scaled[:split]
valid_X = X_scaled[split:]

results = {}

for states in [2, 3, 4]:

    temp_model = hmm.GaussianHMM(
        n_components=states,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )

    temp_model.fit(train_X)

    score = temp_model.score(valid_X) / len(valid_X)

    results[states] = score

    print(f"{states} Hidden States: {score:.4f}")

# Select the model with the best validation score
best_states = max(results, key=results.get)

print("Selected hidden states:", best_states)

# Train final model on complete normal data
model = hmm.GaussianHMM(
    n_components=best_states,
    covariance_type="full",
    n_iter=200,
    random_state=42
)

model.fit(X_scaled)

print("Final HMM Model trained successfully!")


# Calculate training likelihood
train_scores = []

for i in range(0, len(X_scaled) - 50, 50):
    window = X_scaled[i:i + 50]
    score = model.score(window) / len(window)
    train_scores.append(score)

# Set anomaly threshold
threshold = np.percentile(train_scores, 5)

print("Anomaly threshold calculated:", threshold)


# Load test sensor data
test_data = pd.read_csv("data/test_data.csv")

# Scale test data using training scaler
test_X = test_data[["temperature", "vibration"]].values
test_scaled = scaler.transform(test_X)

# Sliding window settings
window_size = 50
step = 10

# Detect anomalies
anomaly_mask = np.zeros(len(test_scaled), dtype=bool)
anomaly_count = 0

for i in range(0, len(test_scaled) - window_size + 1, step):

    window = test_scaled[i:i + window_size]
    score = model.score(window) / len(window)

    if score < threshold:
        anomaly_mask[i:i + window_size] = True
        anomaly_count += 1

print("Anomaly detection completed!")
print("Anomalous windows:", anomaly_count)
print("Anomalous readings:", anomaly_mask.sum())


import matplotlib.pyplot as plt

# Plot sensor temperature
plt.figure(figsize=(14, 6))

plt.plot(
    test_data["temperature"],
    label="Machine Temperature",
    color="blue"
)

# Highlight anomaly regions
plt.fill_between(
    range(len(test_data)),
    test_data["temperature"].min() - 5,
    test_data["temperature"].max() + 5,
    where=anomaly_mask,
    color="red",
    alpha=0.3,
    label="Detected Anomaly"
)

plt.title("HMM Based Machine Anomaly Detection")
plt.xlabel("Time / Sensor Readings")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)

plt.savefig("anomaly_detection.png")
plt.show()

print("Anomaly graph saved successfully!")


# Real-time sensor stream simulation

print("\nStarting Real-Time Anomaly Detection...")

stream_window = 50
stream_step = 10

for i in range(0, len(test_scaled) - stream_window + 1, stream_step):

    current_window = test_scaled[i:i + stream_window]

    score = model.score(current_window) / len(current_window)

    if score < threshold:
        print(f"ALERT! Anomaly detected at reading {i}")
    else:
        print(f"Normal reading at position {i}")

print("Real-Time Monitoring Completed!")
