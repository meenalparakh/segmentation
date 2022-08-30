
import cv2
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
import time
import open3d
from poses.pose_utils import gen_poses, load_data
import sys

import argparse


## Remeber that COLMAP is computing intrinsics on its own

def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.

    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def get_pointcloud_from_depth(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points

def flatten_pcd_data(xyzs, colors, labels=None):

    if labels is None:
        labels = [None]*len(colors)

    xyzs_lst = []
    colors_lst = []
    labels_lst = []

    H, W = xyzs[-1].shape[:2]
    # for xyz, color, label in zip(xyzs, colors, labels):
    for xyz, color, label in zip(xyzs, colors, labels):
        if label is None:
            label = np.zeros((H, W), dtype=np.uint8)
        xyzs_lst.extend(list(xyz.reshape([H*W, 3])))
        colors_lst.extend(list(color.reshape([H*W, 3])))
        labels_lst.extend(list(label.reshape([H*W, 1])))


    return xyzs_lst, colors_lst, labels_lst

def get_pcds(colors, depths, configs, labels=None):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    xyzs = []

    for depth, config in zip(depths, configs):
        # print(f'inside reconstruct heightmap: {color.shape, depth.shape}')
        intrinsics = np.array(config['intrinsics']).reshape(3, 3)
        xyz = get_pointcloud_from_depth(depth, intrinsics)
        position = np.array(config['position']).reshape(3, 1)
        rotation = R.from_quat(config['rotation']).as_matrix()
        # rotation = p.getMatrixFromQuaternion(config['rotation'])
        # rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        xyzs.append(xyz)

    xyzs, colors, labels = flatten_pcd_data(xyzs, colors, labels)
    return xyzs, colors, labels

def visualize_pcd(xyzs, colors):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(xyzs)
    point_cloud.colors = open3d.utility.Vector3dVector(colors)
    open3d.visualization.draw_geometries([point_cloud])

def get_data_from_obs(obs):
    print('len of obs:', len(obs))

    colors = []
    depths = []
    configs = []

    for i in range(len(obs)):
        colors.extend(obs[i]['image']['color'])
        depths.extend(obs[i]['image']['depth'])
        configs.extend(obs[i]['configs'])

    return colors, depths, configs

# def wrapper(poses, imgfiles, imgs):
#     colors = []
#     depths = []
#     configs = []
#     for fname in imgfiles:
#
#     pass

def save_data_for_colmap(colors, depths, configs, dirname):

    color_dir = os.path.join(dirname, 'images')
    depth_dir = os.path.join(dirname, 'depths')
    config_dir = os.path.join(dirname, 'configs')

    if not os.path.exists(color_dir):
        os.makedirs(color_dir)
        os.makedirs(depth_dir)
        os.makedirs(config_dir)

    for idx, color in enumerate(colors):
        color_fname = f'color{idx}.png'
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(color_dir, color_fname), color)

    for idx, depth in enumerate(depths):
        depth_scale = 1000
        depth_fname = f'depth{idx}.png'
        sdepth = depth * depth_scale
        cv2.imwrite(os.path.join(depth_dir, depth_fname),
                    sdepth.astype(np.uint16))

    for idx, config in enumerate(configs):
        config_fname = f'config{idx}.pkl'
        with open(os.path.join(config_dir, config_fname), 'wb') as f:
            pickle.dump(config, f)


if __name__ == '__main__':

    filename = '/Users/meenalp/Desktop/MEng/segmentation/example.pkl'
    with open(filename, 'rb') as f:
        obs = pickle.load(f)
    colors, depths, configs = get_data_from_obs(obs)

    colmap_dir = '/Users/meenalp/Desktop/MEng/segmentation/segmentation/scene_dir'
    # save_data_for_colmap(colors[:2], depths[:2], configs[:2], colmap_dir)
    save_data_for_colmap(colors, depths, configs, colmap_dir)

    success = gen_poses(colmap_dir, 'exhaustive_matcher',
                call_colmap='/Applications/COLMAP.app/Contents/MacOS/colmap')
    if success:
        print('COLMAP ran successfully!')

    # fwh = [1, args.width, args.height]
    # print('factor/width/height args:', fwh)
    # if args.factor is None and args.width is None and args.height is None:
    fwh = [None, None, None]
    poses, bds, imgs, imgfiles = load_data(colmap_dir, *fwh)

    print("Poses:", poses, poses.shape)
    # return


    # xyzs, colors, labels = get_pcds(colors, depths, configs)
    #
    # xyzs = np.array(xyzs)
    # colors = np.array(colors)/255.0
    #
    # visualize_pcd(xyzs, colors)
