import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load your CSV
df = pd.read_csv('./Data/Session/imu_data.csv')

# Convert timestamp from nanoseconds (or milliseconds) to seconds
df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0  # assuming ms

# Fill missing values with 0 (naive, but simple)
df[['acc_x', 'acc_y', 'acc_z']] = df[['acc_x', 'acc_y', 'acc_z']].fillna(0)

# Convert to numpy arrays
t = df['timestamp'].values
acc_x = df['acc_x'].values
acc_y = df['acc_y'].values
acc_z = df['acc_z'].values

# Compute time deltas
dt = np.diff(t, prepend=t[0])

# Integrate acceleration to velocity
vel_x = np.cumsum(acc_x * dt)
vel_y = np.cumsum(acc_y * dt)
vel_z = np.cumsum(acc_z * dt)

# Integrate velocity to position
pos_x = np.cumsum(vel_x * dt)
pos_y = np.cumsum(vel_y * dt)
pos_z = np.cumsum(vel_z * dt)
val = [pos_x, pos_y, pos_z]
# Plot the 2D Path (e.g., X-Y)
for i in range(0, len(val)):
    for n in range(0, len(val)):
        plt.figure(figsize=(10, 6))
        plt.plot(val[i], val[n], marker='o', linestyle='-', color='blue')
        plt.title("Estimated Path from Accelerometer")
        plt.xlabel("Position X")
        plt.ylabel("Position Y")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
