import numpy as np
import open3d
import sys


def main():
    cloud = open3d.io.read_point_cloud("out.ply") # Read the point cloud
    open3d.visualization.draw_geometries([cloud]) # Visualize the point cloud
    sys.exit()
    open3d.visualization.draw_geometries([cloud])  # Visualize the point cloud
    open3d.visualization.draw_geometries([cloud])  # Visualize the point cloud
    open3d.visualization.draw_geometries([cloud]) # Visualize the point cloud


if __name__ == "__main__":
    main()