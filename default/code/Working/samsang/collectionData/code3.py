import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ahrs.filters import Complementary

# Load data
imu_data = pd.read_csv('imu_data.csv')
gps_data = pd.read_csv('gps_data.csv')

# Filter GPS data for accuracy (e.g., < 20m)
gps_data = gps_data[gps_data['accuracy'] < 20]

# Plot 2D path using IMU positions
plt.figure(figsize=(10, 6))
plt.plot(imu_data['position_x'], imu_data['position_y'], '-o', label='Path')
plt.scatter(0, 0, c='red', s=100, label='Start (0,0,0)')
plt.xlabel('X (meters, East)')
plt.ylabel('Y (meters, North)')
plt.title('Traveled Path')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig('path_2d.png')
plt.close()

# Compute orientation using IMU data
acc = imu_data[['acc_x', 'acc_y', 'acc_z']].values
gyro = imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values
mag = imu_data[['mag_x', 'mag_y', 'mag_z']].values
timestamps = imu_data['timestamp'].values
dt = np.diff(timestamps) / 1000.0  # Convert ms to seconds

filter = Complementary(dt=dt.mean())
orientations = []
q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion
for a, g, m in zip(acc, gyro, mag):
    q = filter.update(q, acc=a, gyr=g, mag=m)
    orientations.append(q)

# Convert quaternions to yaw (heading)
from ahrs.common.orientation import q2euler

yaws = [np.degrees(q2euler(q)[2]) for q in orientations]

# Plot path with orientation (downsample for clarity)
downsample = 10
plt.figure(figsize=(10, 6))
plt.plot(imu_data['position_x'], imu_data['position_y'], '-', label='Path')
plt.quiver(
    imu_data['position_x'][::downsample],
    imu_data['position_y'][::downsample],
    np.cos(np.radians(yaws[::downsample])),
    np.sin(np.radians(yaws[::downsample])),
    scale=20,
    color='blue',
    label='Orientation'
)
plt.scatter(0, 0, c='red', s=100, label='Start (0,0,0)')
plt.xlabel('X (meters, East)')
plt.ylabel('Y (meters, North)')
plt.title('Path with Orientation')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig('path_with_orientation.png')
plt.show()
plt.close()

# Optional: 3D plot

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(imu_data['position_x'], imu_data['position_y'], imu_data['position_z'], '-o', label='Path')
ax.scatter(0, 0, 0, c='red', s=100, label='Start (0,0,0)')
ax.set_xlabel('X (meters, East)')
ax.set_ylabel('Y (meters, North)')
ax.set_zlabel('Z (meters, Up)')
ax.set_title('3D Traveled Path')
ax.legend()
plt.savefig('path_3d.png')
plt.show()
plt.close()
