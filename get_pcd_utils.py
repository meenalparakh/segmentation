from __future__ import print_function

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
# from airobot.utils.common import to_rot_mat, rot2quat

def get_pcd_from_depth(cam_intr, cam_extr, rgb_image, depth_image):

    depth_image = 0.032*depth_image
    cam_int_mat_inv = np.linalg.inv(cam_intr)
    H, W = rgb_image.shape[:2]

    img_pixs = np.mgrid[0: H,
                        0: W].reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
    _uv_one = np.concatenate((img_pixs,
                              np.ones((1, img_pixs.shape[1]))))
    uv_one_in_cam = np.dot(cam_int_mat_inv, _uv_one)


    rgb_im = rgb_image
    depth_im = depth_image
    rgb = rgb_im.reshape(-1, 3)
    depth = depth_im.reshape(-1)
    
    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    cam_ext_mat = cam_extr
    pts_in_cam = np.concatenate((pts_in_cam,
                                np.ones((1, pts_in_cam.shape[1]))),
                                axis=0)
    pts_in_world = np.dot(cam_ext_mat, pts_in_cam)
    pcd_pts = pts_in_world[:3, :].T
    pcd_rgb = rgb
    return pcd_pts, pcd_rgb


def get_pcds(colors, depths, configs):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    xyzs_list = []
    colors_list = []

    cam_extrs = []

    print('inside get pcd', len(colors))
    for color, depth, config in zip(colors, depths, configs):
        # print(f'inside reconstruct heightmap: {color.shape, depth.shape}')
        cam_intr = np.array(config['intrinsics']).reshape(3, 3)

        position = np.array(config['position']).reshape(3, 1)
        w, x, y, z = config['orientation']
        rotation = R.from_quat([x, y, z, w]).as_matrix()
        cam_extr = np.eye(4)
        cam_extr[:3,:3] = rotation.T
        cam_extr[:3,3:] = -rotation.T @ position

        xyz, color = get_pcd_from_depth(cam_intr, cam_extr, color, depth)

        xyzs_list.append(xyz)
        colors_list.append(color)
        cam_extrs.append(cam_extr)

    xyzs = np.concatenate(xyzs_list, axis=0)
    colors = np.concatenate(colors_list, axis=0)
    print(xyzs.shape, colors.shape)
    # visualize_poses(transforms)
    
    assert (xyzs.shape[1]==3)
    assert (colors.shape[1]==3)

    # xyzs, colors, labels = flatten_pcd_data(xyzs, colors, labels)
    return xyzs, colors, cam_extrs
