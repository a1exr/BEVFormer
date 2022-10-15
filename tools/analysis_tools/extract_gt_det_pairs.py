# Based on https://github.com/nutonomy/nuscenes-devkit
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from cProfile import label
from tkinter import N, image_names
import mmcv
import argparse
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
import csv
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample, boxes_to_sensor 
from datetime import datetime
from random import shuffle, seed
from scipy.stats import circmean, circvar
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams

categories = ['car', 'pedestrian', 'truck', 'trailer', 'bus', 'motorcycle', 'bicycle']


class Box3D:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None, s=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.h = h      # height
        self.w = w      # width
        self.l = l      # length
        self.ry = ry    # orientation
        self.s = s      # detection score
        self.corners_3d_cam = None

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s)

    @classmethod
    def array2bbox_raw(cls, data):

        bbox = Box3D()
        bbox.w, bbox.l, bbox.h = data.wlh
        bbox.x, bbox.y, bbox.z = data.center
        bbox.ry = data.orientation.angle
        bbox.s = data.score

        return bbox

    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h, bbox.s])


def get_bev_boxes(sample_token, det_list, pose_record, cs_record):
    det_annotations = EvalBoxes()
    det_annotations.add_boxes(sample_token, det_list)
    # Get BEV boxes.
    boxes_det_global = det_annotations[sample_token]
    boxes_det = boxes_to_sensor(boxes_det_global, pose_record, cs_record)
    boxes_det = add_score(boxes_det, boxes_det_global)

    boxes_det = process_dets(boxes_det)
    return boxes_det


def calc_instance_mean_and_var(boxes):
    instance_array = np.array([[cur_box.x, cur_box.y, cur_box.ry, cur_box.l, cur_box.w]  for cur_box in boxes])
    instance_mean = np.mean(instance_array, axis=0)
    instance_var = np.var(instance_array, axis=0)

    # circular mean and variance for theta
    instance_mean[2] = circmean(instance_array[:, 2], low=-np.pi, high=np.pi)
    instance_var[2] = circvar(instance_array[:, 2], low=-np.pi, high=np.pi)

    return instance_array, instance_mean, instance_var


def add_score(dets, global_dets):
    for det, global_det in zip(dets, global_dets):
        det.score = global_det.detection_score

    return dets


def process_dets(dets):
    # convert each detection into the class Box3D

    dets_new = []
    for det in dets:
        det_tmp = Box3D.array2bbox_raw(det)
        dets_new.append(det_tmp)

    return dets_new


def dist3d_bottom(bbox1, bbox2):	
	# Compute distance of bottom center in 3D space, considering the difference in height / 2

	c1 = Box3D.bbox2array(bbox1)[:3]
	c2 = Box3D.bbox2array(bbox2)[:3]
	dist = np.linalg.norm(c1 - c2)

	return dist


def plot_diff_graphs(diff_arrays, plot_name):
    res_type = ['Detection', 'Tracking']
    fig, ax = plt.subplots(4, 2, sharex=True)
    fig.set_figheight(24)
    fig.set_figwidth(40)
    # ax = 8 * [None]
    
    for ind, diff_arr in enumerate(diff_arrays):
        frames = range(diff_arr.shape[0])
        ax[0, ind].plot(frames, diff_arr[:, 0], color='blue',linewidth=2, label='x')
        ax[0, ind].plot(frames, diff_arr[:, 1], color='magenta',linewidth=2, label='y')
        ax[0, ind].set_title(f'Diff = GT - {res_type[ind]}', fontsize = 28)
        ax[0, ind].set_ylabel('X,Y Diff [m]', fontsize = 24)
        ax[0, ind].legend(fontsize=20)
        ax[0, ind].grid(True, which='both')
        ax[0, ind].tick_params(axis='both', which='major', labelsize=20)
        ax[0, ind].minorticks_on()

        ax[1, ind].plot(frames, diff_arr[:, 2], color='black',linewidth=2)
        ax[1, ind].set_ylabel('$\Theta$ Diff [rad]', fontsize = 24)
        ax[1, ind].grid(True, which='both')
        ax[1, ind].tick_params(axis='both', which='major', labelsize=20)
        ax[1, ind].minorticks_on()

        ax[2, ind].plot(frames, diff_arr[:, 3], color='gray',linewidth=2, label='length')
        ax[2, ind].plot(frames, diff_arr[:, 4], color='orange',linewidth=2, label='width')
        ax[2, ind].set_ylabel('L,W Diff [m]', fontsize = 24)
        ax[2, ind].legend(fontsize=20)
        ax[2, ind].grid(True, which='both')
        ax[2, ind].tick_params(axis='both', which='major', labelsize=20)
        ax[2, ind].minorticks_on()

        ax[3, ind].plot(frames, diff_arr[:, 5], color='red',linewidth=2)
        ax[3, ind].set_ylabel(f'{res_type[ind]} Score', fontsize = 24)
        ax[3, ind].set_xlabel('Frame index', fontsize = 24)
        ax[3, ind].grid(True, which='both')
        ax[3, ind].tick_params(axis='both', which='major', labelsize=20)
        ax[3, ind].minorticks_on()

    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()
    
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize BEV results')
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes/trainval')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--results_dir', help='the dir where the results jsons are')
    args = parser.parse_args()

    return args


def main(args, nusc):
    dt = datetime.now()
    str_dt = dt.strftime("%d-%m-%Y_%H:%M:%S")
    header = ['sample_token', 'gt_x', 'gt_z', 'gt_theta', 'gt_l', 'gt_w',
    'det_x', 'det_z', 'det_theta', 'det_l', 'det_w', 'det_score'
    'trk_x', 'trk_z', 'trk_theta', 'trk_l', 'trk_w', 'trk_score']
    empty_line = [None] * len(header)

    bevformer_det_results = mmcv.load(os.path.join(args.results_dir, 'detect_results_nusc.json'))
    bevformer_track_results = mmcv.load(os.path.join(args.results_dir, 'track_results_nusc.json'))
    sample_token_list = list(bevformer_det_results['results'].keys())

    for instance in nusc.instance:
        if instance['nbr_annotations'] < 10:
            continue
        instance_token = instance['token']
        first_annotation_token = instance['first_annotation_token']
        last_annotation_token = instance['last_annotation_token']
        annotation_token = first_annotation_token
        content = nusc.get('sample_annotation', first_annotation_token)
        sample_token = content['sample_token']
        if sample_token not in sample_token_list:
            continue
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_name = nusc.get('scene', scene_token)['name']

        category = [cat for cat in categories if cat in content['category_name']]
        if category == []:
            continue
        category = category[0]

        df = pd.DataFrame(columns=['sample_token', 'gt', 'det', 'trk'])
        for ii in range(instance['nbr_annotations']):
            sample_rec = nusc.get('sample', sample_token)
            sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

            # GT
            tmp_gt = []
            tmp_gt.append(DetectionBox(
                sample_token=sample_token,
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=category,
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=''))

            gt = get_bev_boxes(sample_token, tmp_gt, pose_record, cs_record)[0]

            # Detection
            bbox_anns = bevformer_det_results['results'][sample_token]
            nominees_det_list = []
            for cur_bbox in bbox_anns:
                if cur_bbox['detection_name'] == category and cur_bbox['detection_score'] > 0.3:
                    nominees_det_list.append(DetectionBox(
                        sample_token=sample_token,
                        translation=tuple(cur_bbox['translation']),
                        size=tuple(cur_bbox['size']),
                        rotation=tuple(cur_bbox['rotation']),
                        velocity=tuple(cur_bbox['velocity']),
                        ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in cur_bbox
                        else tuple(cur_bbox['ego_translation']),
                        num_pts=-1 if 'num_pts' not in cur_bbox else int(cur_bbox['num_pts']),
                        detection_name=cur_bbox['detection_name'],
                        detection_score=-1.0 if 'detection_score' not in cur_bbox else float(cur_bbox['detection_score']),
                        attribute_name=cur_bbox['attribute_name']))

            det = []
            if nominees_det_list != []:
                dets = get_bev_boxes(sample_token, nominees_det_list, pose_record, cs_record)
                dists = np.array([dist3d_bottom(gt, det) for det in dets])
                min_dist = np.min(dists)
                th = max([1, 1.2 * min_dist]) if min_dist < 5 else 0
                min_dist_indices = np.where(dists < th)[0]
                det = [dets[min_dist_index] for min_dist_index in min_dist_indices]

            # Tracking
            bbox_anns = bevformer_track_results['results'][sample_token]
            nominees_trk_list = []
            for cur_bbox in bbox_anns:
                tracking_score = -1.0 if 'tracking_score' not in cur_bbox else float(cur_bbox['tracking_score'])
                if cur_bbox['tracking_name'] == category and tracking_score > 0.4:
                    nominees_trk_list.append(DetectionBox(
                        sample_token=sample_token,
                        translation=tuple(cur_bbox['translation']),
                        size=tuple(cur_bbox['size']),
                        rotation=tuple(cur_bbox['rotation']),
                        velocity=tuple(cur_bbox['velocity']),
                        ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in cur_bbox
                        else tuple(cur_bbox['ego_translation']),
                        num_pts=-1 if 'num_pts' not in cur_bbox else int(cur_bbox['num_pts']),
                        detection_name=cur_bbox['tracking_name'],
                        detection_score=tracking_score,
                        attribute_name=cur_bbox['attribute_name']))

            trk = []
            if nominees_trk_list != []:
                trks = get_bev_boxes(sample_token, nominees_trk_list, pose_record, cs_record)
                dists = np.array([dist3d_bottom(gt, trk) for trk in trks])
                min_dist = np.min(dists)
                th = max([1, 1.2 * min_dist]) if min_dist < 5 else 0
                min_dist_indices = np.where(dists < th)[0]
                trk = [trks[min_dist_index] for min_dist_index in min_dist_indices]

            df.loc[ii] = [sample_token, gt, det, trk]

            if annotation_token == last_annotation_token:
                break
            annotation_token = content['next']
            content = nusc.get('sample_annotation', annotation_token)
            sample_token = content['sample_token']

        if len(df) == 0:
            continue

        gt_det_diff = np.full((instance['nbr_annotations'], 6), np.nan)
        gt_trk_diff = np.full((instance['nbr_annotations'], 6), np.nan)
        ind = 0

        save_dir = os.path.join(args.results_dir, 'var_csv', str_dt, category, scene_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_name = os.path.join(save_dir, f'{instance_token}.png')
        csv_name = os.path.join(save_dir, f'{instance_token}.csv')
        with open(csv_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for _, row in df.iterrows():
                gt_results_line = empty_line.copy()
                gt_results_line[:6] = [row['sample_token'], row['gt'].x, row['gt'].y, row['gt'].ry, row['gt'].l, row['gt'].w]

                det_len = len(row['det'])
                trk_len = len(row['trk'])
                max_len = max([det_len, trk_len])
                if max_len == 0:
                    writer.writerow(gt_results_line)
                    ind += 1
                    continue
                for i in range(max_len):
                    results_line = gt_results_line.copy()
                    if i < det_len:
                        results_line[6:12] = [row['det'][i].x, row['det'][i].y, row['det'][i].ry, row['det'][i].l, row['det'][i].w, row['det'][i].s]
                        if i == 0:
                            gt_det_diff[ind, :5] = np.array(results_line[1:6]) - np.array(results_line[6:11])
                            gt_det_diff[ind, 5] = results_line[11]
                    if i < trk_len:
                        results_line[12:] = [row['trk'][i].x, row['trk'][i].y, row['trk'][i].ry, row['trk'][i].l, row['trk'][i].w, row['trk'][i].s]
                        if i == 0:
                            gt_trk_diff[ind, :5] = np.array(results_line[1:6]) - np.array(results_line[12:17])
                            gt_trk_diff[ind, 5] = results_line[17]
                    writer.writerow(results_line)
                ind += 1

        plot_diff_graphs((gt_det_diff, gt_trk_diff), plot_name)


if __name__ == '__main__':
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    main(args, nusc)