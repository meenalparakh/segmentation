from __future__ import print_function

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d
# from airobot.utils.common import to_rot_mat, rot2quat


def draw_geometries(pcds):
    open3d.visualization.draw_geometries(pcds)

# def get_o3d_FOR(origin=[0, 0, 0],size=10):
#     """ 
#     Create a FOR that can be added to the open3d point cloud
#     """
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#     size=size)
#     mesh_frame.translate(origin)
#     return(mesh_frame)

def vector_magnitude(vec):
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def calculate_zy_rotation_for_arrow(vec):
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale=10):
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = open3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None, color=[1,0,0]):
    scale = 1.
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)

    return(mesh)

def save_pcd(xyzs, colors, cam_extrs, fname, save_pcd=True, visualize_pcd=False):
    xyzs = np.array(xyzs)
    colors = np.array(colors)/255.0
    scale=1.0

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(xyzs)
    point_cloud.colors = open3d.utility.Vector3dVector(colors)

    geoms_lst = [point_cloud]
    # geoms_lst = []

    def get_transformed_pt(X, pt):
        x, y, z = pt
        new_pt = X @ np.array([[scale*x],[scale*y],[scale*z],[1.0]])
        return list(new_pt[:3,0])

    # len(cam_extrs)
    for cam_idx in range(len(cam_extrs)):
        cam_extr = cam_extrs[cam_idx]
        cam_origin = get_transformed_pt(cam_extr, [0,0,0])
        cam_x = get_transformed_pt(cam_extr, [1,0,0])
        cam_y = get_transformed_pt(cam_extr, [0,1,0])
        cam_z = get_transformed_pt(cam_extr, [0,0,1])
        arrow_x = get_arrow(origin=cam_origin, end=cam_x, color=[1,0,0])
        arrow_y = get_arrow(origin=cam_origin, end=cam_y, color=[0,1,0])
        arrow_z = get_arrow(origin=cam_origin, end=cam_z, color=[0,0,1])
        geoms_lst.extend([arrow_x, arrow_y, arrow_z])

    if save_pcd:
        open3d.io.write_point_cloud(fname, point_cloud)
    if visualize_pcd:
        open3d.visualization.draw_geometries(geoms_lst)

    print('Point cloud saved.')
    return point_cloud
    
def visualize_poses(transforms):
    positions = np.array([transform[:3, 3] for transform in transforms])
    colors = np.zeros_like(positions)

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(positions)
    point_cloud.colors = open3d.utility.Vector3dVector(colors)

    open3d.visualization.draw([point_cloud], point_size=5)