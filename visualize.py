import numpy as np
import open3d
import os


def main():
    point_file = "/local-scratch/data/shapenet-pcl/03001627/points/1ac6531a337de85f2f7628d6bf38bcc4.pts"
    points = np.loadtxt(point_file).astype(np.float32)
    print(np.asarray(points))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.asarray(points))

    print("hello")
    print(pcd)

    open3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":
    main()
