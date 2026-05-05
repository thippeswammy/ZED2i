import numpy as np
import pandas as pd
from ahrs.filters import Complementary

# Load your CSV
df = pd.read_csv('./Data/Session/imu_data.csv')

acc = imu_data[['acc_x', 'acc_y', 'acc_z']].values
gyro = imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values
timestamps = imu_data['timestamp'].values
dt = np.diff(timestamps) / 1000.0  # Convert ms to seconds

# Apply complementary filter
filter = Complementary(dt=dt.mean())
orientations = []
q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion
for a, g in zip(acc, gyro):
    q = filter.update(q, acc=a, gyr=g)
    orientations.append(q)

# Convert quaternions to Euler angles if needed
