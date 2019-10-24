import numpy as np
import open3d
import os


def visualize_pcl(is_array, point_array=None, point_file=None):
    pcd = open3d.geometry.PointCloud()

    if not is_array:

        # point_file = "/local-scratch/data/shapenet-pcl/03001627/points/1ac6531a337de85f2f7628d6bf38bcc4.pts"
        points = np.loadtxt(point_file).astype(np.float32)
        print(np.asarray(points))
        pcd.points = open3d.utility.Vector3dVector(np.asarray(points))

    else:
        pcd.points = open3d.utility.Vector3dVector(np.asarray(point_array))

    print(pcd)

    open3d.visualization.draw_geometries([pcd])





