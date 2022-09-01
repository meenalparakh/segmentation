from __future__ import print_function

import cv2
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
import time
import open3d
from poses.pose_utils import gen_poses, load_data
from poses.colmap_read_model import read_cameras_text
import sys
# from airobot.utils.common import to_rot_mat, rot2quat

import argparse
from airobot import Robot
import shutil
import json, random
from datetime import datetime
import argparse
from visualization_utils import *
from get_pcd_utils import *

def collect_data(n, colmap_dir, depth_scale=1000):

    color_dir = os.path.join(colmap_dir, 'images')
    depth_dir = os.path.join(colmap_dir, 'depths')
    config_dir = os.path.join(colmap_dir, 'configs')

    if not os.path.exists(color_dir):
        os.makedirs(color_dir)
        os.makedirs(depth_dir)
        os.makedirs(config_dir)

    # print('Starting!')
    robot = Robot('ur5e_2f140',
                  pb=False,
                  use_arm=False,
                  use_cam=True)
    cam_params_shared = None

    idx = 0
    # while True:
    for idx in range(n):
        input('Press any key')
        # if keyboard.is_pressed("q"):
        #     print('exiting data collection')
        #     break

        # time.sleep(2)
        color, depth = robot.cam.get_images(get_rgb=True, get_depth=True)

        intrinsic = robot.cam.get_cam_int()
        fname = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(idx,':', fname)

        # pos, ori = get_pose_from_transformation(robot.cam.get_cam_ext())
        config = {'intrinsics': intrinsic}
                #   'position': position,
                #   'rotation': ori}

        cam_params = get_cam_params_from_intr(intrinsic)
        if cam_params_shared is None:
            cam_params_shared = cam_params
        else:
            assert (cam_params_shared == cam_params)

        color_fname = fname + '.png'
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(color_dir, color_fname), color)

        depth_fname = fname + '.pkl'
        with open(os.path.join(depth_dir, depth_fname), 'wb') as f:
            pickle.dump(depth, f)

        config_fname = fname + '.pkl'
        with open(os.path.join(config_dir, config_fname), 'wb') as f:
            pickle.dump(config, f)
        
        idx += 1

    with open(os.path.join(colmap_dir, 'shared_intrinsic.pkl'), 'wb') as f:
        pickle.dump(cam_params_shared, f)


def clear_colmap_dirs(colmap_dir):
    sparse = os.path.join(colmap_dir, 'sparse')
    database = os.path.join(colmap_dir, 'database.db')
    out = os.path.join(colmap_dir, 'colmap_output.txt')

    if os.path.exists(sparse):
        shutil.rmtree(sparse)

    if os.path.exists(database):
        os.remove(database)
        os.remove(out)

def get_cam_params_from_intr(intr):
    fx = intr[0,0]
    fy = intr[1,1]
    cx = intr[0,2]
    cy = intr[1,2]
    params = [fx, fy, cx, cy]
    return ','.join([str(p) for p in params])

def get_intr_from_cam_params(basedir):
    cameras = read_cameras_text(os.path.join(basedir, 'sparse/0/cameras.txt'))
    assert (len(cameras)==1)
    cam = cameras[list(cameras.keys())[0]]
    fx, fy, cx, cy = cam.params
    h, w = cam.height, cam.width
    intr = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])
    return intr


def read_all(basedir, depth_scale=1000):
    fwh = [None, None, None]
    poses, imgfiles = load_data(colmap_dir, *fwh)
    cam_intrinsics = get_intr_from_cam_params(basedir)

    print(f"Cam intrinsic: {cam_intrinsics}")
    print("Poses length:", len(poses))

    colors = []
    depths = []
    configs = []

    for idx in range(len(imgfiles)):
        print(f'Image: {imgfiles[idx]}')
        # print(f'Camera pose: {poses[idx]}')
        # print('----------------------------\n')
        color_fname = os.path.join(basedir, 'images', imgfiles[idx])
        depth_fname = os.path.join(basedir, 'depths', imgfiles[idx].replace('.png', '.pkl'))
        config_fname = os.path.join(basedir, 'configs', imgfiles[idx].replace('.png', '.pkl'))

        with open(depth_fname, 'rb') as f:
            depth = pickle.load(f)

        color = cv2.imread(color_fname)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        with open(config_fname, 'rb') as f:
            config = pickle.load(f)
            pos, ori = poses[idx]
            config['position'] = pos
            config['orientation'] = ori
            config['intrinsics'] = cam_intrinsics
            # config['position'] = [0,0,0]
            # config['rotation'] = [0,0,0,1]
            print(f'File: {imgfiles[idx]}, {config}')

        colors.append(color)
        depths.append(depth)
        configs.append(config)

    last = len(colors)
    print(last, 'inside read al')
    return colors[:last], depths[:last], configs[:last]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-data-collection', action='store_true', default=False)
    parser.add_argument('--skip-colmap', action='store_true', default=False)
    parser.add_argument('--dense-reconstruct', action='store_true', default=False)
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()

    colmap_dir = './scene_dir'

    if not args.skip_data_collection:
        print('Collecting data ...')
        if os.path.exists(colmap_dir):
            shutil.rmtree(colmap_dir)

        collect_data(args.n, colmap_dir)

    if not args.skip_colmap:
        clear_colmap_dirs(colmap_dir)
        with open(os.path.join(colmap_dir, 'shared_intrinsic.pkl'), 'rb') as f:
            cam_params_shared = pickle.load(f)  
        success = gen_poses(colmap_dir, 'exhaustive_matcher', cam_model='PINHOLE', 
                    cam_params=cam_params_shared,
                    call_colmap='colmap', dense_reconstruct=args.dense_reconstruct)
        if success:
            print('COLMAP ran successfully!')

    colors, depths, configs = read_all(colmap_dir)
    print('inside main', len(colors))
    xyzs, colors, cam_extrs = get_pcds(colors, depths, configs)
    save_pcd(xyzs, colors, cam_extrs, 
            os.path.join(colmap_dir, 'final_merged_pcd.ply'), 
            save_pcd=True,
            visualize_pcd=True)