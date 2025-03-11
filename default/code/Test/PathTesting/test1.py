import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load IMU Data
file_path = "../collectionData/Data/ZED2i_Data/imu_data.csv"  # Update the path as needed

imu_data = pd.read_csv("../collectionData/Data/ZED2i_Data/imu_data.csv")

# Extract timestamps & acceleration
timestamps = imu_data["timestamp"].values
acc_y = imu_data["acceleration_x"].values
acc_x = imu_data["acceleration_y"].values
acc_z = imu_data["acceleration_z"].values

# Convert timestamps to time intervals
dt = np.diff(timestamps, prepend=timestamps[0])

# Integrate acceleration to get velocity
vel_x = np.cumsum(acc_x * dt)
vel_y = np.cumsum(acc_y * dt)
vel_z = np.cumsum(acc_z * dt)

# Integrate velocity to get position
pos_x = np.cumsum(vel_x * dt)
pos_y = np.cumsum(vel_y * dt)
pos_z = np.cumsum(vel_z * dt)

# Plot the path in 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_x, pos_y, pos_z, marker='o', linestyle='-', color='b', label="Path")

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("3D Path from IMU Data")
ax.legend()

plt.show()
