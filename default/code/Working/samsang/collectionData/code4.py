import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------
# Step 1: Load GPS and IMU Data
# ----------------------------------------------

# Load your data (Make sure to have gps_data.csv and imu_data.csv ready)
gps_data = pd.read_csv(
    'gps_data.csv')  # timestamp, latitude, longitude, altitude, speed, bearing, accuracy, local_x, local_y, local_z
imu_data = pd.read_csv('imu_data.csv')  # timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z

# Sort timestamps just in case
gps_data = gps_data.sort_values('timestamp').reset_index(drop=True)
imu_data = imu_data.sort_values('timestamp').reset_index(drop=True)

# ----------------------------------------------
# Step 2: Interpolate GPS Positions to IMU Timestamps
# ----------------------------------------------

gps_interp = {
    'local_x': np.interp(imu_data['timestamp'], gps_data['timestamp'], gps_data['local_x']),
    'local_y': np.interp(imu_data['timestamp'], gps_data['timestamp'], gps_data['local_y']),
    'local_z': np.interp(imu_data['timestamp'], gps_data['timestamp'], gps_data['local_z']),
}

# ----------------------------------------------
# Step 3: Estimate Orientation using Gyroscope
# ----------------------------------------------

dt = np.diff(imu_data['timestamp'], prepend=imu_data['timestamp'][0])

rotations = [R.identity()]  # Start with no rotation

for i in range(1, len(imu_data)):
    gx, gy, gz = imu_data.loc[i, ['gyro_x', 'gyro_y', 'gyro_z']].values
    omega = np.array([gx, gy, gz])  # rad/s
    theta = np.linalg.norm(omega * dt[i])  # rotation angle
    if theta > 0:
        axis = omega / np.linalg.norm(omega)
        delta_rotation = R.from_rotvec(axis * theta)
    else:
        delta_rotation = R.identity()

    rotations.append(rotations[-1] * delta_rotation)

# ----------------------------------------------
# Step 4: Rotate Accelerations into World Frame
# ----------------------------------------------

acc_world = []

for i in range(len(imu_data)):
    acc_body = imu_data.loc[i, ['acc_x', 'acc_y', 'acc_z']].values
    acc_world_i = rotations[i].apply(acc_body)
    acc_world.append(acc_world_i)

acc_world = np.array(acc_world)

# ----------------------------------------------
# Step 5: Remove Gravity
# ----------------------------------------------

gravity = np.array([0, 0, 9.81])  # m/s^2 (assuming Z is up)
acc_world_no_gravity = acc_world - gravity

# ----------------------------------------------
# Step 6: Integrate Acceleration to get Velocity and Position
# ----------------------------------------------

velocity = np.zeros((len(imu_data), 3))
position = np.zeros((len(imu_data), 3))

for i in range(1, len(imu_data)):
    velocity[i] = velocity[i - 1] + acc_world_no_gravity[i - 1] * dt[i]
    position[i] = position[i - 1] + velocity[i - 1] * dt[i] + 0.5 * acc_world_no_gravity[i - 1] * dt[i] ** 2

# ----------------------------------------------
# Step 7: Simple GPS + IMU Sensor Fusion
# ----------------------------------------------

alpha = 0.98  # IMU trust factor, (1-alpha) GPS trust factor

fused_position = alpha * position + (1 - alpha) * np.column_stack(
    (gps_interp['local_x'], gps_interp['local_y'], gps_interp['local_z']))

# ----------------------------------------------
# Step 8: Plot the Trajectories
# ----------------------------------------------

plt.figure(figsize=(10, 8))
plt.plot(position[:, 0], position[:, 1], label='IMU Only Trajectory', linestyle='--')
plt.plot(gps_interp['local_x'], gps_interp['local_y'], label='GPS Only Trajectory', linestyle=':')
plt.plot(fused_position[:, 0], fused_position[:, 1], label='Fused Trajectory', linewidth=2)
plt.legend()
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('Trajectory Reconstruction using Sensor Fusion')
plt.grid(True)
plt.axis('equal')
plt.show()

# ----------------------------------------------
# End of Code
# ----------------------------------------------
