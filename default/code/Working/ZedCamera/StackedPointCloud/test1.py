import open3d as o3d
import os

# Define the folder containing the point cloud files
pcd_folder = "../collectionData/Data/ZED2i_Data/PointCloud"
output_file = "../collectionData/Data/ZED2i_Data/StackedPointCloud.ply"

# List all PLY files in the directory
pcd_files = sorted([f for f in os.listdir(pcd_folder) if f.endswith(".ply")])

# Initialize an empty global point cloud
global_pcd = o3d.geometry.PointCloud()

# Load and merge each point cloud
for file in pcd_files:
    file_path = os.path.join(pcd_folder, file)
    pcd = o3d.io.read_point_cloud(file_path)
    global_pcd += pcd  # Append to the global point cloud
    print(f"✅ Loaded {file}")

# Save the stacked point cloud
o3d.io.write_point_cloud(output_file, global_pcd)
print(f"📂 Stacked Point Cloud saved as '{output_file}'")
