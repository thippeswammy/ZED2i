import pandas as pd
import numpy as np
import open3d as o3d

# Load IMU Data
file_path = "../collectionData/Data/ZED2i_Data/imu_data.csv"  # Update the path as needed
data = pd.read_csv(file_path)

# Normalize timestamp
data["timestamp"] = data["timestamp"] - data["timestamp"].iloc[0]
time = data["timestamp"].values

# Extract IMU data
ax = data["acceleration_x"].values
ay = data["acceleration_y"].values
az = data["acceleration_z"].values

# Compute time differences (dt)
dt = np.diff(time, prepend=time[0])

# Initialize position and velocity
positions = np.zeros((len(time), 3))
velocities = np.zeros((len(time), 3))

# Integrate acceleration → velocity → position
for i in range(1, len(time)):
    velocities[i] = velocities[i-1] + np.array([ax[i], ay[i], az[i]]) * dt[i]
    positions[i] = positions[i-1] + velocities[i] * dt[i]

# Create Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create Base Plane (XZ Plane at Y = 0)
plane = o3d.geometry.TriangleMesh.create_box(width=10, height=0.01, depth=10)
plane.translate([-5, -0.01, -5])  # Move plane down slightly
plane.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray color
vis.add_geometry(plane)

# Add grid (Coordinate Frame)
grid = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
vis.add_geometry(grid)

# Create 3D Line Set
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(positions)
lines = [[i, i+1] for i in range(len(positions)-1)]
line_set.lines = o3d.utility.Vector2iVector(lines)
vis.add_geometry(line_set)

# Live Update
for i in range(len(positions)):
    line_set.points = o3d.utility.Vector3dVector(positions[:i+1])
    vis.update_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()

vis.run()
vis.destroy_window()
