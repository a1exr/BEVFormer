# Based on https://github.com/nutonomy/nuscenes-devkit
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

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
    save_dir = os.path.join(args.results_dir, 'var_csv', str_dt)
    os.makedirs(save_dir, exist_ok=True)

    header = ['scene_name', 'initial_x', 'initial_z', 'number_of_annotations',
    'x_mean', 'z_mean', 'theta_mean', 'l_mean', 'w_mean',
    'x_var', 'z_var', 'theta_var', 'l_var', 'w_var']

    bevformer_det_results = mmcv.load(os.path.join(args.results_dir, 'detect_results_nusc.json'))
    sample_token_list = list(bevformer_det_results['results'].keys())

    for category in categories:
        csv_name = os.path.join(save_dir, f'{category}.csv')
        with open(csv_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        if first_sample_token in sample_token_list:
            scene_name = scene['name']
            st_index = sample_token_list.index(first_sample_token)
            end_index = sample_token_list.index(last_sample_token)

            print('---Scene name: ' + scene_name)
            used_anns = []
            for ref_index in range(st_index, end_index):
                ref_sample_token = sample_token_list[ref_index]

                sample_rec = nusc.get('sample', ref_sample_token)
                sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

                ref_smaple_bbox_anns = bevformer_det_results['results'][ref_sample_token]
                for i, cur_bbox in enumerate(ref_smaple_bbox_anns):
                    if (ref_index, i) in used_anns or cur_bbox['detection_score'] < 0.3:
                        continue
                    category = cur_bbox['detection_name']
                    
                    ref = []
                    ref.append(DetectionBox(
                                sample_token=ref_sample_token,
                                translation=tuple(cur_bbox['translation']),
                                size=tuple(cur_bbox['size']),
                                rotation=tuple(cur_bbox['rotation']),
                                velocity=tuple(cur_bbox['velocity']),
                                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in cur_bbox
                                else tuple(cur_bbox['ego_translation']),
                                num_pts=-1 if 'num_pts' not in cur_bbox else int(cur_bbox['num_pts']),
                                detection_name=category,
                                detection_score=-1.0 if 'detection_score' not in cur_bbox else float(cur_bbox['detection_score']),
                                attribute_name=cur_bbox['attribute_name']))

                    ref_box = get_bev_boxes(ref_sample_token, ref, pose_record, cs_record)[0]

                    bbox_det_list = []
                    bbox_det_list.append(ref_box)
                    # Detection
                    for cur_index in range(ref_index+1, end_index+1):
                        sample_token = sample_token_list[cur_index]

                        sample_rec = nusc.get('sample', sample_token)
                        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
                        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

                        sample_bbox_anns = bevformer_det_results['results'][sample_token]
                        nominees_det_list = []
                        nominees_indices_list = []
                        for j, nominee_bbox in enumerate(sample_bbox_anns):
                            if nominee_bbox['detection_name'] == category and nominee_bbox['detection_score'] > 0.3:
                                nominees_indices_list.append((cur_index, j))
                                nominees_det_list.append(DetectionBox(
                                    sample_token=sample_token,
                                    translation=tuple(nominee_bbox['translation']),
                                    size=tuple(nominee_bbox['size']),
                                    rotation=tuple(nominee_bbox['rotation']),
                                    velocity=tuple(nominee_bbox['velocity']),
                                    ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in nominee_bbox
                                    else tuple(nominee_bbox['ego_translation']),
                                    num_pts=-1 if 'num_pts' not in nominee_bbox else int(nominee_bbox['num_pts']),
                                    detection_name=category,
                                    detection_score=-1.0 if 'detection_score' not in nominee_bbox else float(nominee_bbox['detection_score']),
                                    attribute_name=nominee_bbox['attribute_name']))

                        if nominees_det_list != []:
                            nominees_boxes = get_bev_boxes(sample_token, nominees_det_list, pose_record, cs_record)
                            dists = [dist3d_bottom(ref_box, det_box) for det_box in nominees_boxes]
                            min_dist_index = np.argmin(dists)
                            if dists[min_dist_index] < max(ref_box.l, ref_box.h):
                                bbox_det_list.append(nominees_boxes[min_dist_index])
                                used_anns.append(nominees_indices_list[min_dist_index])
                                ref_box = get_bev_boxes(sample_token, nominees_det_list[min_dist_index:min_dist_index+1], pose_record, cs_record)[0]
                            else:
                                break
                        
                    instance_array, instance_mean, instance_var = calc_instance_mean_and_var(bbox_det_list)
                    seq_len = len(bbox_det_list)

                    if seq_len >= 10:
                        results_line = [scene_name, instance_array[0][0], instance_array[0][1], seq_len,
                        instance_mean[0], instance_mean[1], instance_mean[2], instance_mean[3], instance_mean[4],
                        instance_var[0], instance_var[1], instance_var[2], instance_var[3], instance_var[4]]

                        csv_name = os.path.join(save_dir, f'{category}.csv')
                        with open(csv_name, 'a', encoding='UTF8', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(results_line)


if __name__ == '__main__':
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    main(args, nusc)
