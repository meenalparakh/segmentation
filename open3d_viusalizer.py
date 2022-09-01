import open3d
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='./scene_dir/final_merged_pcd.ply')

    args = parser.parse_args()

    pcd = open3d.io.read_point_cloud(args.fname)
    open3d.visualization.draw_geometries([pcd])