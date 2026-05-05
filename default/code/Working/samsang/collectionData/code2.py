import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Step 1: Load GPS and IMU data
gps_data = pd.read_csv('gps_data.csv')
imu_data = pd.read_csv('imu_data.csv')

# Step 2: Preprocess timestamps
gps_data['timestamp'] = pd.to_datetime(gps_data['timestamp'])
imu_data['timestamp'] = pd.to_datetime(imu_data['timestamp'])

# Step 3: Interpolate IMU to GPS timestamps
imu_data = imu_data.set_index('timestamp')
imu_interp = imu_data.reindex(gps_data['timestamp'], method='nearest').reset_index()

# Step 4: Extract needed fields
latitudes = gps_data['latitude'].values
longitudes = gps_data['longitude'].values
acc_x = imu_interp['acc_x'].values
acc_y = imu_interp['acc_y'].values
acc_z = imu_interp['acc_z'].values


# Step 5: Smooth GPS using IMU Acceleration (simple method)
# Apply low-pass filter to acceleration data to remove noise
def low_pass_filter(data, cutoff=0.1, fs=1.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


acc_x_filtered = low_pass_filter(acc_x)
acc_y_filtered = low_pass_filter(acc_y)

# Integrate acceleration to velocity (simple cumulative sum)
dt = np.median(np.diff(gps_data['timestamp']).astype('timedelta64[ms]').astype(float)) / 1000.0
vel_x = np.cumsum(acc_x_filtered) * dt
vel_y = np.cumsum(acc_y_filtered) * dt

# Further integrate velocity to displacement (position changes)
disp_x = np.cumsum(vel_x) * dt
disp_y = np.cumsum(vel_y) * dt

# Step 6: Create corrected path
# Convert small displacements to latitude/longitude offsets
# Rough approximation: 1 degree latitude ≈ 111,320 meters
meters_per_degree_lat = 111320
meters_per_degree_lon = 40075000 * np.cos(np.radians(latitudes)) / 360

# Calculate corrected lat/lon
corrected_latitudes = latitudes + (disp_y / meters_per_degree_lat)
corrected_longitudes = longitudes + (disp_x / meters_per_degree_lon)

# Step 7: Plot original vs corrected path
plt.figure(figsize=(12, 8))
plt.plot(longitudes, latitudes, label='Original GPS Path', color='blue')
plt.plot(corrected_longitudes, corrected_latitudes, label='Smoothed Path (GPS + IMU)', color='red')
plt.title('Path Reconstruction using GPS and IMU')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
