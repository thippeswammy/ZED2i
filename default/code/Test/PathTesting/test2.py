import numpy as np
import open3d as o3d
import pandas as pd

# Load IMU Data
file_path = "../collectionData/Data/ZED2i_Data/imu_data.csv"  # Update the path as needed
data = pd.read_csv(file_path)

# Normalize timestamp
data["timestamp"] = data["timestamp"] - data["timestamp"].iloc[0]
time = data["timestamp"].values

# Extract acceleration data
ax = data["acceleration_x"].values
ay = data["acceleration_y"].values
az = data["acceleration_z"].values

# Compute time differences
dt = np.diff(time, prepend=time[0])

# Initialize position & velocity
x, y, z = [0], [0], [0]  # Start at origin
vx, vy, vz = [0], [0], [0]

# Integrate Acceleration → Velocity → Position
for i in range(1, len(time)):
    vx.append(vx[-1] + ax[i] * dt[i])
    vy.append(vy[-1] + ay[i] * dt[i])
    vz.append(vz[-1] + az[i] * dt[i])

    x.append(x[-1] + vx[-1] * dt[i])
    y.append(y[-1] + vy[-1] * dt[i])
    z.append(z[-1] + vz[-1] * dt[i])

# Convert to numpy arrays
trajectory = np.vstack((x, y, z)).T

# Create Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add Grid
grid = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
vis.add_geometry(grid)

# Create 3D Line Set
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(trajectory)
lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
line_set.lines = o3d.utility.Vector2iVector(lines)
vis.add_geometry(line_set)

# Live Update
for i in range(len(trajectory)):
    line_set.points = o3d.utility.Vector3dVector(trajectory[:i + 1])
    vis.update_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()

vis.run()
vis.destroy_window()
