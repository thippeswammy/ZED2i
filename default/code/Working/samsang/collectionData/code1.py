import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ahrs.common.orientation import q2euler
from ahrs.filters import Complementary

# Load GPS and IMU data
gps_data = pd.read_csv('gps_data.csv')
imu_data = pd.read_csv('imu_data.csv')

# Filter GPS data with reasonable accuracy (e.g., < 20m)
gps_data = gps_data[gps_data['accuracy'] < 20]

# Ensure numeric format for all relevant columns
gps_data['timestamp'] = pd.to_numeric(gps_data['timestamp'], errors='coerce')
imu_data['timestamp'] = pd.to_numeric(imu_data['timestamp'], errors='coerce')

# Drop rows with NaN values after conversion
gps_data = gps_data.dropna(subset=['timestamp'])
imu_data = imu_data.dropna(subset=['timestamp'])

# Extract cleaned data arrays
gps_timestamps = gps_data['timestamp'].values
timestamps = imu_data['timestamp'].values

# Compute local coordinates for GPS data (relative to starting point)
start_lat, start_lon = gps_data.iloc[0][['latitude', 'longitude']]
earth_radius = 6371000  # Earth's radius in meters


def gps_to_local(lat, lon):
    delta_lat = np.radians(lat - start_lat)
    delta_lon = np.radians(lon - start_lon)
    x = earth_radius * delta_lon * np.cos(np.radians(start_lat))
    y = earth_radius * delta_lat
    return x, y


gps_data['local_x'], gps_data['local_y'] = zip(*gps_data.apply(
    lambda row: gps_to_local(row['latitude'], row['longitude']), axis=1))

# Plot 2D path using local coordinates
plt.figure(figsize=(10, 6))
plt.plot(gps_data['local_x'], gps_data['local_y'], '-o', label='GPS Path')
plt.scatter(0, 0, c='red', s=100, label='Start (0,0,0)')
plt.xlabel('X (meters, East)')
plt.ylabel('Y (meters, North)')
plt.title('Traveled Path')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig('path_2d.png')
plt.close()

# Process IMU data for orientation
acc = imu_data[['acc_x', 'acc_y', 'acc_z']].values
gyro = imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values
mag = imu_data[['mag_x', 'mag_y', 'mag_z']].values
timestamps = imu_data['timestamp'].values
dt = np.diff(timestamps) / 1000.0  # Convert ms to seconds

# Apply Complementary Filter for orientation
filter = Complementary(dt=dt.mean())
orientations = []
q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion
for a, g, m in zip(acc, gyro, mag):
    q = filter.update(q, acc=a, gyr=g, mag=m)
    orientations.append(q)

# Convert quaternions to yaw (heading) for visualization
yaws = [np.degrees(q2euler(q)[2]) for q in orientations]

# Interpolate IMU yaw to GPS timestamps
imu_yaws = np.interp(gps_timestamps, timestamps, yaws)

# Plot path with orientation
plt.figure(figsize=(10, 6))
plt.quiver(gps_data['local_x'], gps_data['local_y'],
           np.cos(np.radians(imu_yaws)),
           np.sin(np.radians(imu_yaws)),
           scale=20, color='blue', label='Orientation')
plt.scatter(0, 0, c='red', s=100, label='Start (0,0,0)')
plt.xlabel('X (meters, East)')
plt.ylabel('Y (meters, North)')
plt.title('Path with Orientation')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig('path_with_orientation.png')
plt.close()
