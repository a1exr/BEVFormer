import argparse
from tkinter import OFF
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
from math import ceil
import random
import copy
import pickle
import io

import mmcv
from nuscenes.nuscenes import NuScenes
from torch import rad2deg
from visual import lidar_render

categories = ['Car', 'Pedestrian', 'Truck', 'Trailer', 'Bus', 'Motorcycle', 'Bicycle']
custom_val = \
    ['scene-0003', 'scene-0092', 'scene-0272', 'scene-0521', 'scene-0771', 'scene-0796', 'scene-0914', 'scene-0924',
    'scene-0968', 'scene-1063']


def plot_diff_graphs(gt_instance_df, det_instance_df, trk_instance_df, plot_path, ax):
    gt_df = gt_instance_df[['width', 'length', 'X', 'Y', 'Theta']]
    det_df = det_instance_df[['width', 'length', 'X', 'Y', 'Theta']]
    trk_df = trk_instance_df[['width', 'length', 'X', 'Y', 'Theta']]
    gt_df.reset_index(drop=True, inplace=True)
    det_df.reset_index(drop=True, inplace=True)
    trk_df.reset_index(drop=True, inplace=True)

    det_diff = gt_df-det_df
    det_diff['Theta'] = np.where(det_diff['Theta'].abs() > np.pi , det_diff['Theta'].abs()-2*np.pi, det_diff['Theta'])
    det_diff['Theta'] = np.where(det_diff['Theta'].abs() < -np.pi , 2*np.pi-det_diff['Theta'].abs(), det_diff['Theta'])
    det_diff['Theta'] = det_diff['Theta'] * 180 / np.pi
    trk_diff = gt_df-trk_df
    trk_diff['Theta'] = np.where(trk_diff['Theta'].abs() > np.pi , trk_diff['Theta'].abs()-2*np.pi, trk_diff['Theta'])
    trk_diff['Theta'] = np.where(trk_diff['Theta'].abs() < -np.pi , 2*np.pi-trk_diff['Theta'].abs(), trk_diff['Theta'])
    trk_diff['Theta'] = trk_diff['Theta'] * 180 / np.pi
    diff_dfs = (det_diff, trk_diff)

    det_score = det_instance_df['score']
    trk_score = trk_instance_df['score']
    det_score.reset_index(drop=True, inplace=True)
    trk_score.reset_index(drop=True, inplace=True)
    score_dfs = (det_score, trk_score)

    ax[40].clear()
    gt_df.plot(y=['X', 'Y'], kind='line', ax=ax[40], color='r', linewidth=4, style=['-', '--'])
    det_df.plot(y=['X', 'Y'], kind='line', ax=ax[40], color='k', linewidth=3, style=['-', '--'], label=['__','__'])
    trk_df.plot(y=['X', 'Y'], kind='line', ax=ax[40], color='c', linewidth=2, style=['-', '--'], label=['__','__'])
    ax[40].set_title(f'Actual Values', fontsize = 28)
    # ax[40+i].set_ylabel('X,Y Error [m]', fontsize = 24)

    gt_df['Theta'] = gt_df['Theta'] * 180 / np.pi
    det_df['Theta'] = det_df['Theta'] * 180 / np.pi
    trk_df['Theta'] = trk_df['Theta'] * 180 / np.pi
    ax[41].clear()
    gt_df.plot(y='Theta', kind='line', ax=ax[41], color='r', linewidth=4, label='$\Theta$')
    det_df.plot(y='Theta', kind='line', ax=ax[41], color='k', linewidth=3, label='__')
    trk_df.plot(y='Theta', kind='line', ax=ax[41], color='c', linewidth=2, label='__')
    # ax[41].set_ylabel('$\Theta$ Error [deg]', fontsize = 24)

    ax[42].clear()
    gt_df.plot(y=['length', 'width'], kind='line', ax=ax[42], color='r', linewidth=4, style=['-', '--'])
    det_df.plot(y=['length', 'width'], kind='line', ax=ax[42], color='k', linewidth=3, style=['-', '--'], label=['__','__'])
    trk_df.plot(y=['length', 'width'], kind='line', ax=ax[42], color='c', linewidth=2, style=['-', '--'], label=['__','__'])
    # ax[42].set_ylabel('L,W Error [m]', fontsize = 24)

    res_type = ['Detection', 'Tracking'] 
    for ind, (diff_df, score_df) in enumerate(zip(diff_dfs, score_dfs)):
        i = 4 * ind
        ax[43+i].clear()
        diff_df.plot(y=['X', 'Y'], kind='line', ax=ax[43+i], color=['blue', 'magenta'], linewidth=3)
        ax[43+i].set_title(f'Error = GT - {res_type[ind]}', fontsize = 28)
        # ax[40+i].set_ylabel('X,Y Error [m]', fontsize = 24)

        ax[44+i].clear()
        diff_df.plot(y='Theta', kind='line', ax=ax[44+i], color='black', linewidth=3, label='$\Theta$')
        # ax[44+i].set_ylabel('$\Theta$ Error [deg]', fontsize = 24)

        ax[45+i].clear()
        diff_df.plot(y=['length', 'width'], kind='line', ax=ax[45+i], color=['gray', 'orange'], linewidth=3)
        # ax[45+i].set_ylabel('L,W Error [m]', fontsize = 24)

        ax[46+i].clear()
        score_df.plot(y='score', kind='line', ax=ax[46+i], color='red', linewidth=3, label='confidence')
        # ax[46+i].set_ylabel(f'{res_type[ind]} Score', fontsize = 24)
        # ax[46+i].set_xlabel('Frame index', fontsize = 24)

    trk_instance_df.sig_R_Theta = trk_instance_df.sig_R_Theta * 180 / np.pi
    trk_instance_df.plot(y=['sig_R_X', 'sig_R_Y'], kind='line', ax=ax[43], color=['blue', 'magenta'], linestyle='dashdot', label=['__','__'])
    trk_instance_df.plot(y='sig_R_Theta', kind='line', ax=ax[44], color='black', linestyle='dashdot', label='__')
    trk_instance_df.plot(y=['sig_R_l', 'sig_R_w'], kind='line', ax=ax[45], color=['gray', 'orange'], linestyle='dashdot', label=['__','__'])

    trk_instance_df.sig_R_X = -trk_instance_df.sig_R_X
    trk_instance_df.sig_R_Y = -trk_instance_df.sig_R_Y
    trk_instance_df.sig_R_Theta = -trk_instance_df.sig_R_Theta
    trk_instance_df.sig_R_l = -trk_instance_df.sig_R_l
    trk_instance_df.sig_R_w = -trk_instance_df.sig_R_w

    trk_instance_df.plot(y=['sig_R_X', 'sig_R_Y'], kind='line', ax=ax[43], color=['blue', 'magenta'], linestyle='dashdot', label=['$R\sigma_X$', '$R\sigma_Y$'])
    trk_instance_df.plot(y='sig_R_Theta', kind='line', ax=ax[44], color='black', linestyle='dashdot', label='$R\sigma_\Theta$')
    trk_instance_df.plot(y=['sig_R_l', 'sig_R_w'], kind='line', ax=ax[45], color=['gray', 'orange'], linestyle='dashdot', label=['$R\sigma_l$', '$R\sigma_w$'])

    trk_instance_df.sig_P_Theta = trk_instance_df.sig_P_Theta * 180 / np.pi
    trk_instance_df.plot(y=['sig_P_X', 'sig_P_Y'], kind='line', ax=ax[47], color=['blue', 'magenta'], linestyle='dashdot', label=['__','__'])
    trk_instance_df.plot(y='sig_P_Theta', kind='line', ax=ax[48], color='black', linestyle='dashdot', label='__')
    trk_instance_df.plot(y=['sig_P_l', 'sig_P_w'], kind='line', ax=ax[49], color=['gray', 'orange'], linestyle='dashdot', label=['__','__'])

    trk_instance_df.sig_P_X = -trk_instance_df.sig_P_X
    trk_instance_df.sig_P_Y = -trk_instance_df.sig_P_Y
    trk_instance_df.sig_P_Theta = -trk_instance_df.sig_P_Theta
    trk_instance_df.sig_P_l = -trk_instance_df.sig_P_l
    trk_instance_df.sig_P_w = -trk_instance_df.sig_P_w

    trk_instance_df.plot(y=['sig_P_X', 'sig_P_Y'], kind='line', ax=ax[47], color=['blue', 'magenta'], linestyle='dashdot', label=['$P\sigma_X$', '$P\sigma_Y$'])
    trk_instance_df.plot(y='sig_P_Theta', kind='line', ax=ax[48], color='black', linestyle='dashdot', label='$P\sigma_\Theta$')
    trk_instance_df.plot(y=['sig_P_l', 'sig_P_w'], kind='line', ax=ax[49], color=['gray', 'orange'], linestyle='dashdot', label=['$P\sigma_l$', '$P\sigma_w$'])

    window_size = max([7.5, 2.0*np.mean(gt_df['length'])])
    for i, gt_row in gt_instance_df.iterrows():
        if i > 39 or (ax[i] is None): continue
        if np.isnan(gt_row.frame):
            ax[i].set_visible(False)
        else:
            ax[i].set_visible(True)
            frame_idx = int(gt_row.frame)
            ax[frame_idx].set_xlim((gt_row.X-window_size, gt_row.X+window_size))
            ax[frame_idx].set_ylim((gt_row.Y-window_size, gt_row.Y+window_size))
            # ax[frame_idx].text(0.05, 0.95, frame_idx, horizontalalignment='center', verticalalignment='center', transform=ax[frame_idx].transAxes, color='r', size=20)
            ax[frame_idx].plot(gt_row.X, gt_row.Y, color='r', marker='x', markersize=19, markeredgewidth=3, linestyle='None')
            det_row = det_df.iloc[i]
            if not np.isnan(det_row.X):
                ax[frame_idx].plot(det_row.X, det_row.Y, color='k', marker='+', markersize=19, markeredgewidth=3, linestyle='None')
            trk_row = trk_df.iloc[i]
            if not np.isnan(trk_row.X):
                ax[frame_idx].plot(trk_row.X, trk_row.Y, color='c', marker='+', markersize=19, markeredgewidth=3, linestyle='None')

    for i in range(4):
        ylim_0 = ax[43+i].get_ylim()
        ylim_1 = ax[47+i].get_ylim()
        ylim = (min([ylim_0[0], ylim_1[0]]), max([ylim_0[1], ylim_1[1]]))
        ax[43+i].set_ylim(ylim)
        ax[47+i].set_ylim(ylim)

    cols = [2, 1, 2, 4, 2, 4, 1, 4, 2, 4, 1]
    for i in range(40, 51, 1):
        ax[i].set_xlim(0, 41)
        ax[i].legend(fontsize=24, ncol=cols[i-40])
        ax[i].grid(True, which='both')
        ax[i].minorticks_on()

    plt.tight_layout()
    plt.savefig(plot_path)
    # plt.close()

    for i, gt_row in gt_instance_df.iterrows():
        if i > 39: continue
        if not np.isnan(gt_row.frame):
            frame_idx = int(gt_row.frame)

            trk_row = trk_df.iloc[i]
            if not np.isnan(trk_row.X):
                ax[frame_idx].lines.pop(-1)
                ax[frame_idx].lines[-1].remove()

            det_row = det_df.iloc[i]
            if not np.isnan(det_row.X):
                ax[frame_idx].lines.pop(-1)
                ax[frame_idx].lines[-1].remove()

            ax[frame_idx].lines.pop(-1)
            ax[frame_idx].lines[-1].remove()

    return


def parse_args():
    parser = argparse.ArgumentParser(description='Plot GT vs. Det vs. Trk results')
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--det_res_dir', type=str, required=True)
    parser.add_argument('--trk_res_dir', type=str, required=True)
    parser.add_argument('--results_dir', type=str, help='the dir where the results jsons are', required=True)
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes/trainval')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--amount', type=int, default=30, help='number of scenes')
    parser.add_argument('--random', action='store_true', help='pick scenes randomly')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--custom_val', action='store_true', help='custom scenes list')
    parser.add_argument('--instances_rate', type=float, default=0.1, help='percentage of instances to run (0:1)')
    args = parser.parse_args()

    return args


def main(args, nusc):
    header = ['Frame', 'ID', 'GT_X', 'GT_Y', 'GT_Theta', 'GT_length', 'GT_width',
    'Det_X', 'Det_Y', 'Det_Theta', 'Det_length', 'Det_width', 'Det_score',
    'Trk_X', 'Trk_Y', 'Trk_Theta', 'Trk_length', 'Trk_width', 'Trk_score']
    empty_line = [None] * len(header)
    norm_th = 5

    gt_files = os.listdir(args.gt_dir)
    det_folders = os.listdir(args.det_res_dir)
    trk_folders = os.listdir(args.trk_res_dir)

    random.seed(args.seed)
    dt = datetime.now()
    extension = '_rand' if args.random else ''
    str_dt = dt.strftime("%d-%m-%Y_%H:%M:%S") + extension
    timed_dir = os.path.join(args.results_dir, 'error_stats', str_dt)

    bevformer_det_results = mmcv.load(os.path.join(args.results_dir, 'detect_results_nusc.json'))
    bevformer_track_results = mmcv.load(os.path.join(args.results_dir, 'track_results_nusc.json'))
    sample_token_list = list(bevformer_det_results['results'].keys())

    scene_count = 0
    scenes_list = [scene for scene in nusc.scene]
    if args.random:
        random.shuffle(scenes_list)
    for scene in scenes_list:
        scene_name = scene['name']
        if args.custom_val and scene_name not in custom_val: continue
        scene_file_name = scene_name + '.txt'
        if scene_file_name not in gt_files: continue
        if scene_count == args.amount: break

        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        if first_sample_token in sample_token_list:
            print('---Scene name ' + scene_name)
            scene_dir = os.path.join(timed_dir, scene_name)
            st_index = sample_token_list.index(first_sample_token)
            end_index = sample_token_list.index(last_sample_token)

            fig = plt.figure()
            fig.set_figheight(20)
            fig.set_figwidth(40)
            ax = 51 * [None]
            for i, sample_token in enumerate(sample_token_list[st_index:end_index+1]):
                if i == 40: break
                ax_loc = (3, i) if i < 4 else (4+(i-4)//12, (i-4)%12)
                ax[i] = plt.subplot2grid(shape=(7, 12), loc=ax_loc, colspan=1, rowspan=1)
                lidar_render(nusc, sample_token, bevformer_det_results, bevformer_track_results, ax[i])
                ax[i].text(0.05, 0.95, i, horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes, color='r', size=20)
                # ax[i].tick_params(direction='in')
            scene_count += 1

            ax[40] = plt.subplot2grid(shape=(7, 12), loc=(0, 0), colspan=4)
            ax[41] = plt.subplot2grid(shape=(7, 12), loc=(1, 0), colspan=4, sharex=ax[40])
            ax[42] = plt.subplot2grid(shape=(7, 12), loc=(2, 0), colspan=4, sharex=ax[41])
            for ii in range(0, 5, 4):
                ax[43+ii] = plt.subplot2grid(shape=(7, 12), loc=(0, 4+ii), colspan=4)
                ax[44+ii] = plt.subplot2grid(shape=(7, 12), loc=(1, 4+ii), colspan=4, sharex=ax[43+ii])
                ax[45+ii] = plt.subplot2grid(shape=(7, 12), loc=(2, 4+ii), colspan=4, sharex=ax[44+ii])
                ax[46+ii] = plt.subplot2grid(shape=(7, 12), loc=(3, 4+ii), colspan=4, sharex=ax[45+ii])

            for ii in range(40, 51, 1):
                ax[ii].tick_params(axis='both', which='major', labelsize=20)
                ax[ii].grid(True, which='major', linewidth=3)
                ax[ii].grid(True, which='minor', linewidth=1)
                ax[ii].minorticks_on()
                major_ticks = np.arange(0, 41, 5)
                minor_ticks = np.arange(0, 41, 1)
                ax[ii].set_xticks(major_ticks)
                ax[ii].set_xticks(minor_ticks, minor = True)

                ax[ii].legend(fontsize=24, loc='lower right')
        
            cur_scene_gt_file_path = os.path.join(args.gt_dir, scene_file_name)
            cur_scene_gt_df = pd.read_csv(cur_scene_gt_file_path, sep=' ', header=None, usecols=[0,1,2,11,12,13,15,16],\
                names=['frame', 'ID', 'cat', 'width', 'length', 'X', 'Y', 'Theta'])
            for category in categories:
                print(f'------- {category}')
                cur_gt_df = cur_scene_gt_df[cur_scene_gt_df.cat == category]

                det_flag = True
                det_folder = [cur_folder for cur_folder in det_folders if category in cur_folder]
                if det_folder == []: det_flag = False
                det_folder_path = os.path.join(args.det_res_dir, det_folder[0], scene_file_name)
                if not os.path.isfile(det_folder_path): det_flag = False

                if det_flag:
                    cur_det_df = pd.read_csv(det_folder_path, sep=',', header=None, usecols=[0,6,8,9,10,12,13],\
                        names=['frame', 'score', 'width', 'length', 'X', 'Y', 'Theta'])
                    cur_det_df = cur_det_df[cur_det_df.score >= 0.3]

                trk_flag = True
                trk_folder = [cur_folder for cur_folder in trk_folders if category in cur_folder]
                if trk_folder == []: trk_flag = False
                trk_folder_path = os.path.join(args.trk_res_dir, trk_folder[0], 'graphs_info', scene_file_name)
                if not os.path.isfile(trk_folder_path): trk_flag = False

                if trk_flag:
                    cur_trk_df = pd.read_csv(trk_folder_path, sep=' ', header=None, usecols=[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],\
                        names=['frame', 'ID', 'score', 'width', 'length', 'X', 'Y', 'Theta',\
                            'sig_P_X', 'sig_P_Y', 'sig_P_Theta', 'sig_P_l', 'sig_P_w',
                            'sig_R_X', 'sig_R_Y', 'sig_R_Theta', 'sig_R_l', 'sig_R_w'])
                    # cur_trk_df = cur_trk_df[cur_trk_df.score >= 0.4]
                
                instances = cur_gt_df.ID.unique()
                instances_len = len(instances)
                inst_size = ceil(args.instances_rate * instances_len)
                inst_size = min(instances_len, 3) if inst_size < 3 else inst_size
                chosen_instances = np.random.choice(instances, size=inst_size, replace=False)
                for instance in chosen_instances:
                    print(f'- Instance #{instance} -')
                    save_dir = os.path.join(scene_dir, category)
                    os.makedirs(save_dir, exist_ok=True)
                    plot_path = os.path.join(save_dir, f'{category}_{instance}.png')
                    csv_name = os.path.join(save_dir, f'{category}_{instance}.csv')
                    with open(csv_name, 'w', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)

                        cur_instance_gt_df = cur_gt_df[cur_gt_df.ID == instance]
                        if len(cur_instance_gt_df) < 10: continue
                        gt_instance_df = pd.DataFrame(columns=['frame', 'width', 'length', 'X', 'Y', 'Theta'], index=range(41))

                        if det_flag:
                            det_instance_df = pd.DataFrame(columns=['frame', 'score', 'width', 'length', 'X', 'Y', 'Theta'], index=range(41))
                        if trk_flag:
                            trk_instance_df = pd.DataFrame(columns=['frame', 'score', 'width', 'length', 'X', 'Y', 'Theta',\
                            'sig_P_X', 'sig_P_Y', 'sig_P_Theta', 'sig_P_l', 'sig_P_w',\
                                'sig_R_X', 'sig_R_Y', 'sig_R_Theta', 'sig_R_l', 'sig_R_w'], index=range(41))
                        
                        instance_id = None
                        last_norms = np.ones(5) * norm_th
                        for _, cur_gt_row in cur_instance_gt_df.iterrows():
                            results_line = empty_line.copy()
                            results_line[:7] = [cur_gt_row['frame'], cur_gt_row['ID'],\
                                cur_gt_row['X'], cur_gt_row['Y'], cur_gt_row['Theta'], cur_gt_row['length'], cur_gt_row['width']]

                            frame_ind = cur_gt_row.frame
                            cur_gt = cur_gt_row[['width', 'length', 'X', 'Y', 'Theta']]
                            gt_instance_df.loc[frame_ind] = cur_gt_row.copy()

                            # Detection
                            if det_flag:
                                min_value = norm_th
                                relevant_det_df = cur_det_df[cur_det_df.frame == frame_ind]
                                if not relevant_det_df.empty:
                                    cur_det = relevant_det_df[['width', 'length', 'X', 'Y', 'Theta']]

                                    diff = -cur_det.subtract(cur_gt, axis=1)
                                    diff['Theta'] = np.where(diff['Theta'].abs() > np.pi , 2*np.pi-diff['Theta'].abs(), diff['Theta'])
                                    diff_norm = diff.pow(2).sum(1).pow(0.5)
                                    min_value = diff_norm.min()
                                    weighted_last_norms = [1, 2, 3, 4, 5] * last_norms
                                    cur_th = max(2.0, 1.5 * np.mean(weighted_last_norms) / 3.0)
                                    if min_value < cur_th:
                                        cur_det_row = relevant_det_df.loc[diff_norm.idxmin()]
                                        results_line[7:13] = [cur_det_row['X'], cur_det_row['Y'], cur_det_row['Theta'],\
                                            cur_det_row['length'], cur_det_row['width'], cur_det_row['score']]
                                        det_instance_df.loc[frame_ind] = cur_det_row.copy()

                                min_value = min(norm_th, min_value)
                                last_norms[:4] = last_norms[1:]
                                last_norms[4] = min_value

                            # Tracking
                            if trk_flag:
                                if instance_id is None:
                                    relevant_trk_df = cur_trk_df[cur_trk_df.frame == frame_ind]
                                    if not relevant_trk_df.empty:
                                        cur_trk = relevant_trk_df[['width', 'length', 'X', 'Y', 'Theta']]
                                        diff = -cur_trk.subtract(cur_gt, axis=1)
                                        diff['Theta'] = np.where(diff['Theta'].abs() > np.pi , 2*np.pi-diff['Theta'].abs(), diff['Theta'])
                                        diff_norm = diff.pow(2).sum(1).pow(0.5)
                                        min_value = diff_norm.min()
                                        if min_value < norm_th:
                                            cur_trk_row = relevant_trk_df.loc[diff_norm.idxmin()]
                                            instance_id = cur_trk_row['ID']
                                            results_line[13:] = [cur_trk_row['X'], cur_trk_row['Y'], cur_trk_row['Theta'],\
                                                cur_trk_row['length'], cur_trk_row['width'], cur_trk_row['score']]
                                            trk_instance_df.loc[frame_ind] = cur_trk_row.copy()
                                else:
                                    relevant_trk_df = cur_trk_df[(cur_trk_df.frame == frame_ind) & (cur_trk_df.ID == instance_id)]
                                    if len(relevant_trk_df) == 1:
                                        cur_trk_row = relevant_trk_df.iloc[0]
                                        results_line[13:] = [cur_trk_row['X'], cur_trk_row['Y'], cur_trk_row['Theta'],\
                                            cur_trk_row['length'], cur_trk_row['width'], cur_trk_row['score']]
                                        trk_instance_df.loc[frame_ind] = cur_trk_row.copy()
                            
                            writer.writerow(results_line)

                    plot_diff_graphs(gt_instance_df, det_instance_df, trk_instance_df, plot_path, ax)

                    # plt.tight_layout()
                    # plt.savefig(plot_path)
                    # plt.close()


if __name__ == '__main__':
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    main(args, nusc)
