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
from nuscenes.eval.detection.render import visualize_sample
from datetime import datetime
from random import shuffle, seed

cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK_RIGHT',
 'CAM_BACK',
 'CAM_BACK_LEFT',
 'CAM_FRONT_LEFT']

detections = ['car',
 'pedestrian',
 'bicycle',
 'motorcycle',
 'bus',
 'trailer',
 'truck']

custom_val = \
    ['scene-0003', 'scene-0092', 'scene-0272', 'scene-0521', 'scene-0771', 'scene-0796', 'scene-0914', 'scene-0924',
    'scene-0968', 'scene-1063']

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams


def create_video(save_dir, scene_name):
    output_dir = os.path.join(save_dir, 'videos')
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f'{scene_name}.avi')
    if os.path.isfile(video_path):
        os.remove(video_path)

    scene_path = os.path.join(save_dir, scene_name)
    all_frames = sorted([os.path.join(scene_path, cur_img) for cur_img in os.listdir(scene_path)])
    # all_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    height, width, channels = cv2.imread(all_frames[0]).shape
    out = cv2.VideoWriter(video_path, fourcc, 2, (width, height), True)

    print('Creating Video Animation')
    for filename in tqdm(all_frames):
        img = cv2.imread(filename)
        out.write(img)
    out.release()


def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = 'render.png',
        extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record['data'][cam] for cam in select_cams]
    print('bbox in cams:', select_cams)
    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # Plot CAMERA view.
    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)


def get_sample_data(sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens=None,
                    use_flat_vehicle_coordinates: bool = False):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def lidar_render(nusc, sample_token, det_data, track_data, ax=None):
    bbox_gt_list = []
    bbox_det_list = []
    bbox_track_list = []
    # GT
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        category_name = category_to_detection_name(content['category_name'])
        if category_name in detections:
            try:
                bbox_gt_list.append(DetectionBox(
                    sample_token=content['sample_token'],
                    translation=tuple(content['translation']),
                    size=tuple(content['size']),
                    rotation=tuple(content['rotation']),
                    velocity=nusc.box_velocity(content['token'])[:2],
                    ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                    else tuple(content['ego_translation']),
                    num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                    detection_name=category_name,
                    detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                    attribute_name=''))
            except:
                pass

    # Detection
    bbox_anns = det_data['results'][sample_token]
    for content in bbox_anns:
        category_name = content['detection_name']
        detection_score = -1.0 if 'detection_score' not in content else float(content['detection_score'])
        if category_name in detections and detection_score > 0.3:
            bbox_det_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=tuple(content['velocity']),
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=category_name,
                detection_score=detection_score,
                attribute_name=content['attribute_name']))

    # Tracking
    bbox_anns = track_data['results'][sample_token]
    for content in bbox_anns:
        tracking_score = -1.0 if 'tracking_score' not in content else float(content['tracking_score'])
        if tracking_score > 0.4:
            bbox_track_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=tuple(content['velocity']),
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=content['tracking_name'],
                detection_score=tracking_score,
                attribute_name=content['attribute_name']))

    gt_annotations = EvalBoxes()
    det_pred_annotations = EvalBoxes()
    track_pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    det_pred_annotations.add_boxes(sample_token, bbox_det_list)
    track_pred_annotations.add_boxes(sample_token, bbox_track_list)
    # print('green is ground truth')
    # print('blue is the predited result')
    # visualize_sample(nusc, sample_token, gt_annotations, pred_annotations, savepath=out_path + '_bev')
    visualize_sample(nusc, sample_token, gt_annotations, det_pred_annotations, track_pred_annotations, ax)


def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def render_sample_data(
        nusc: NuScenes,
        sample_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        # new_scene: bool = False,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        det_data=None,
        track_data=None,
      ) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    sample = nusc.get('sample', sample_token)
    timestamp = str(sample['timestamp'])
    # sample = data['results'][sample_token_list[0]][0]
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    # split_name = None
    # image_name = None
    # scene_name = None

    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(40)
    ax = 7 * [None]
    
    ax[0] = plt.subplot2grid(shape=(4, 8), loc=(0, 0), colspan=4, rowspan=4)
    lidar_render(nusc, sample_token, det_data, track_data, ax[0])

    ax[1] = plt.subplot2grid(shape=(4, 8), loc=(1, 4), colspan=2)
    ax[2] = plt.subplot2grid(shape=(4, 8), loc=(0, 5), colspan=2)
    ax[3] = plt.subplot2grid(shape=(4, 8), loc=(1, 6), colspan=2)
    ax[4] = plt.subplot2grid(shape=(4, 8), loc=(2, 4), colspan=2)
    ax[5] = plt.subplot2grid(shape=(4, 8), loc=(3, 5), colspan=2)
    ax[6] = plt.subplot2grid(shape=(4, 8), loc=(2, 6), colspan=2)
    
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]

        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            assert False
        elif sensor_modality == 'camera':
            # Load boxes and image.
            boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                        name=record['detection_name'], token='predicted') for record in
                    det_data['results'][sample_token] if record['detection_score'] > 0.3 and record['detection_name'] in detections]
            data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token,
                                                                         box_vis_level=box_vis_level, pred_anns=boxes)
            _, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)

            data = Image.open(data_path)

            # Show image.
            ax[ind+1].imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes_gt:
                    c = np.array(get_color(box.name)) / 255.0
                    box.render(ax[ind+1], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax[ind+1].set_xlim(0, data.size[0])
            ax[ind+1].set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax[ind+1].axis('off')

    print('Rendering sample: %s' % sample_token)
    ax[0].set_title(f'sample_token: {sample_token}', fontsize=24)
    ax[2].set_title('GT', fontsize=24, color='r', fontweight='bold')
    # ax[8].set_title('Pred', fontsize=24, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'{timestamp}_{sample_token}'))
    # if verbose:
    #     plt.show()
    plt.close()
    
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize BEV results')
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes/trainval')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--results_dir', help='the dir where the results jsons are')
    parser.add_argument('--amount', type=int, default=30, help='number of samples / scenes')
    parser.add_argument('--scene_video', action='store_true', help='visualize entire scene')
    parser.add_argument('--random', action='store_true', help='pick samples randomly')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--custom_val', action='store_true', help='custom scenes list')
    args = parser.parse_args()

    return args


def main(args, nusc):
    seed(args.seed)
    dt = datetime.now()
    extension = '_rand' if args.random else ''
    str_dt = dt.strftime("%d-%m-%Y_%H:%M:%S") + extension
    save_dir = os.path.join(args.results_dir, 'plots', str_dt)
    os.makedirs(save_dir, exist_ok=True)

    bevformer_det_results = mmcv.load(os.path.join(args.results_dir, 'detect_results_nusc.json'))
    bevformer_track_results = mmcv.load(os.path.join(args.results_dir, 'track_results_nusc.json'))
    sample_token_list = list(bevformer_det_results['results'].keys())

    if args.scene_video:
        i = 0
        scenes_list = [scene for scene in nusc.scene]
        if args.random:
            shuffle(scenes_list)
        for scene in scenes_list:
            if i == args.amount:
                break
            scene_name = scene['name']
            if args.custom_val and scene_name not in custom_val: continue
            print('---Scene name ' + scene_name)
            scene_dir = os.path.join(save_dir, scene_name)

            # new_scene = True
            first_sample_token = scene['first_sample_token']
            last_sample_token = scene['last_sample_token']
            if first_sample_token in sample_token_list:
                os.makedirs(scene_dir, exist_ok=True)
                st_index = sample_token_list.index(first_sample_token)
                end_index = sample_token_list.index(last_sample_token)

                for sample_token in sample_token_list[st_index:end_index+1]:
                    render_sample_data(nusc, sample_token, det_data=bevformer_det_results, track_data=bevformer_track_results, out_path=scene_dir)
                    # new_scene = False

                create_video(save_dir, scene_name)
                i += 1

    else:
        if args.random:
            shuffle(sample_token_list)
        for sample_token in sample_token_list[:args.amount]:
            render_sample_data(nusc, sample_token, det_data=bevformer_det_results, track_data=bevformer_track_results, out_path=save_dir)


if __name__ == '__main__':
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    main(args, nusc)
