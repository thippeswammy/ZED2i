import matplotlib.pyplot as plt
import pandas as pd

# Load IMU data
imu_data = pd.read_csv("../collectionData/Data/ZED2i_Data4/imu_data.csv")
# Extract X and Y positions
pos_x = imu_data["position_x"].values
pos_z = imu_data["position_z"].values

# Plot the top-view trajectory
plt.figure(figsize=(10, 6))
plt.plot(pos_x, pos_z, marker="o", linestyle="-", markersize=2, label="Traveled Path")
plt.xlabel("X Position (m)")
plt.ylabel("Z Position (m)")
plt.title("Top View of Traveled Path")
plt.legend()
plt.grid()
plt.axis("equal")  # Ensures correct aspect ratio
plt.show()
