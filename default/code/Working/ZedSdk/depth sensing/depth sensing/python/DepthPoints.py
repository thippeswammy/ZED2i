import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import argparse

width = 720
height = 404
point_cloud_file = f"Pointcloud.ply"
print(point_cloud_file)

point_cloud = sl.Mat(width, height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
point_cloud.read(point_cloud_file)

width = point_cloud.get_width()
height = point_cloud.get_height()

img = [[]]
for i in range(width):
    for n in range(height):
        img[i][n] = point_cloud.get_value(10, 10)

print(img)
