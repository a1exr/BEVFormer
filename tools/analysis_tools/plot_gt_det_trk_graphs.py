import argparse
from tkinter import OFF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime

categories = ['Car', 'Pedestrian', 'Truck', 'Trailer', 'Bus', 'Motorcycle', 'Bicycle']
R_matrix = np.array([[0.25,0,0,0,0],  	# x  
                     [0,1.8,0,0,0],  	# z
                     [0,0,1,0,0], 	    # theta
                     [0,0,0,0.8,0],	    # l
                     [0,0,0,0,0.1]]) 	# w


def plot_diff_graphs(gt_instance_df, det_instance_df, trk_instance_df, plot_path):
    gt_df = gt_instance_df[['width', 'length', 'X', 'Y', 'Theta']]
    det_df = det_instance_df[['width', 'length', 'X', 'Y', 'Theta']]
    trk_df = trk_instance_df[['width', 'length', 'X', 'Y', 'Theta']]
    gt_df.reset_index(drop=True, inplace=True)
    det_df.reset_index(drop=True, inplace=True)
    trk_df.reset_index(drop=True, inplace=True)
    det_diff = gt_df-det_df
    det_diff['Theta'] = np.where(det_diff['Theta'].abs() > np.pi , det_diff['Theta'].abs()-2*np.pi, det_diff['Theta'])
    det_diff['Theta'] = np.where(det_diff['Theta'].abs() < -np.pi , 2*np.pi-det_diff['Theta'].abs(), det_diff['Theta'])
    trk_diff = gt_df-trk_df
    trk_diff['Theta'] = np.where(trk_diff['Theta'].abs() > np.pi , trk_diff['Theta'].abs()-2*np.pi, trk_diff['Theta'])
    trk_diff['Theta'] = np.where(trk_diff['Theta'].abs() < -np.pi , 2*np.pi-trk_diff['Theta'].abs(), trk_diff['Theta'])
    diff_dfs = (det_diff, trk_diff)

    det_score = det_instance_df['score']
    trk_score = trk_instance_df['score']
    det_score.reset_index(drop=True, inplace=True)
    trk_score.reset_index(drop=True, inplace=True)
    score_dfs = (det_score, trk_score)

    res_type = ['Detection', 'Tracking']
    fig, ax = plt.subplots(4, 2, sharex=True)
    fig.set_figheight(24)
    fig.set_figwidth(40)
    
    for ind, (diff_df, score_df) in enumerate(zip(diff_dfs, score_dfs)):
        diff_df.plot(y=['X', 'Y'], kind='line', ax=ax[0, ind], color=['blue', 'magenta'], linewidth=2)
        ax[0, ind].set_title(f'Error = GT - {res_type[ind]}', fontsize = 28)
        ax[0, ind].set_ylabel('X,Y Error [m]', fontsize = 24)
        ax[0, ind].grid(True, which='both')
        ax[0, ind].tick_params(axis='both', which='major', labelsize=20)
        ax[0, ind].minorticks_on()

        diff_df.plot(y='Theta', kind='line', ax=ax[1, ind], color='black', linewidth=2, label='$\Theta$')
        ax[1, ind].set_ylabel('$\Theta$ Error [rad]', fontsize = 24)
        ax[1, ind].grid(True, which='both')
        ax[1, ind].tick_params(axis='both', which='major', labelsize=20)
        ax[1, ind].minorticks_on()

        diff_df.plot(y=['length', 'width'], kind='line', ax=ax[2, ind], color=['gray', 'orange'], linewidth=2)
        ax[2, ind].set_ylabel('L,W Error [m]', fontsize = 24)
        ax[2, ind].grid(True, which='both')
        ax[2, ind].tick_params(axis='both', which='major', labelsize=20)
        ax[2, ind].minorticks_on()

        score_df.plot(kind='line', ax=ax[3, ind], color='red', linewidth=2, legend=False)
        ax[3, ind].set_ylabel(f'{res_type[ind]} Score', fontsize = 24)
        ax[3, ind].set_xlabel('Frame index', fontsize = 24)
        ax[3, ind].grid(True, which='both')
        ax[3, ind].tick_params(axis='both', which='major', labelsize=20)
        ax[3, ind].minorticks_on()

    diag_R = np.sqrt(np.diag(R_matrix))
    samples = len(score_df)
    frames = range(samples)

    ax[0, 0].plot(frames, samples*[diag_R[0]], color='blue', linestyle='dashdot', label='$R\sigma_X$')
    ax[0, 0].plot(frames, samples*[-diag_R[0]], color='blue', linestyle='dashdot')
    ax[0, 0].plot(frames, samples*[diag_R[1]], color='magenta', linestyle='dashdot', label='$R\sigma_Y$')
    ax[0, 0].plot(frames, samples*[-diag_R[1]], color='magenta', linestyle='dashdot')
    ax[0, 0].legend(fontsize=24, loc='lower right')

    ax[1, 0].plot(frames, samples*[diag_R[2]], color='black', linestyle='dashdot', label='$R\sigma_\Theta$')
    ax[1, 0].plot(frames, samples*[-diag_R[2]], color='black', linestyle='dashdot')
    ax[1, 0].legend(fontsize=24, loc='lower right')

    ax[2, 0].plot(frames, samples*[diag_R[3]], color='gray', linestyle='dashdot', label='$R\sigma_l$')
    ax[2, 0].plot(frames, samples*[-diag_R[3]], color='gray', linestyle='dashdot')
    ax[2, 0].plot(frames, samples*[diag_R[4]], color='orange', linestyle='dashdot', label='$R\sigma_w$')
    ax[2, 0].plot(frames, samples*[-diag_R[4]], color='orange', linestyle='dashdot')
    ax[2, 0].legend(fontsize=24, loc='lower right')

    trk_instance_df.plot(y=['sig_X', 'sig_Y'], kind='line', ax=ax[0, 1], color=['blue', 'magenta'], linestyle='dashdot', label=['__','__'])
    trk_instance_df.plot(y='sig_Theta', kind='line', ax=ax[1, 1], color='black', linestyle='dashdot', label='__')
    trk_instance_df.plot(y=['sig_l', 'sig_w'], kind='line', ax=ax[2, 1], color=['gray', 'orange'], linestyle='dashdot', label=['__','__'])

    trk_instance_df.sig_X = -trk_instance_df.sig_X
    trk_instance_df.sig_Y = -trk_instance_df.sig_Y
    trk_instance_df.sig_Theta = -trk_instance_df.sig_Theta
    trk_instance_df.sig_l = -trk_instance_df.sig_l
    trk_instance_df.sig_w = -trk_instance_df.sig_w

    trk_instance_df.plot(y=['sig_X', 'sig_Y'], kind='line', ax=ax[0, 1], color=['blue', 'magenta'], linestyle='dashdot', label=['$P\sigma_X$', '$P\sigma_Y$'])
    ax[0, 1].legend(fontsize=24, loc='lower right')
    trk_instance_df.plot(y='sig_Theta', kind='line', ax=ax[1, 1], color='black', linestyle='dashdot', label='$P\sigma_\Theta$')
    ax[1, 1].legend(fontsize=24, loc='lower right')
    trk_instance_df.plot(y=['sig_l', 'sig_w'], kind='line', ax=ax[2, 1], color=['gray', 'orange'], linestyle='dashdot', label=['$P\sigma_l$', '$P\sigma_w$'])
    ax[2, 1].legend(fontsize=24, loc='lower right')

    for i in range(4):
        ylim_0 = ax[i, 0].get_ylim()
        ylim_1 = ax[i, 1].get_ylim()
        ylim = (min([ylim_0[0], ylim_1[0]]), max([ylim_0[1], ylim_1[1]]))
        ax[i, 0].set_ylim(ylim)
        ax[i, 1].set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Plot GT vs. Det vs. Trk results')
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--det_res_dir', type=str, required=True)
    parser.add_argument('--trk_res_dir', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    return args


def main(args):
    dt = datetime.now()
    str_dt = dt.strftime("%d-%m-%Y_%H:%M:%S")
    header = ['Frame', 'ID', 'GT_X', 'GT_Y', 'GT_Theta', 'GT_length', 'GT_width',
    'Det_X', 'Det_Y', 'Det_Theta', 'Det_length', 'Det_width', 'Det_score',
    'Trk_X', 'Trk_Y', 'Trk_Theta', 'Trk_length', 'Trk_width', 'Trk_score']
    empty_line = [None] * len(header)

    gt_files = os.listdir(args.gt_dir)
    det_folders = os.listdir(args.det_res_dir)
    trk_folders = os.listdir(args.trk_res_dir)

    if True:
        plots_path = os.path.join(args.results_dir, 'plots')
        last_plot = sorted(os.listdir(plots_path))[-1]
        scenes_in_last_plot = os.listdir(os.path.join(plots_path, last_plot))
        gt_files = [gt_file for gt_file in gt_files if os.path.splitext(gt_file)[0] in scenes_in_last_plot]
 
    for gt_file in gt_files:
        print(f'*** {gt_file} ***')
        scene_name = os.path.splitext(gt_file)[0]
        cur_scene_gt_file_path = os.path.join(args.gt_dir, gt_file)
        cur_scene_gt_df = pd.read_csv(cur_scene_gt_file_path, sep=' ', header=None, usecols=[0,1,2,11,12,13,15,16],\
            names=['frame', 'ID', 'cat', 'width', 'length', 'X', 'Y', 'Theta'])
        for category in categories:
            print(f'------- {category}')
            cur_gt_df = cur_scene_gt_df[cur_scene_gt_df.cat == category]

            det_flag = True
            det_folder = [cur_folder for cur_folder in det_folders if category in cur_folder]
            if det_folder == []: det_flag = False
            det_folder_path = os.path.join(args.det_res_dir, det_folder[0], gt_file)
            if not os.path.isfile(det_folder_path): det_flag = False

            if det_flag:
                cur_det_df = pd.read_csv(det_folder_path, sep=',', header=None, usecols=[0,6,8,9,10,12,13],\
                    names=['frame', 'score', 'width', 'length', 'X', 'Y', 'Theta'])
                cur_det_df = cur_det_df[cur_det_df.score >= 0.3]

            trk_flag = True
            trk_folder = [cur_folder for cur_folder in trk_folders if category in cur_folder]
            if trk_folder == []: trk_flag = False
            trk_folder_path = os.path.join(args.trk_res_dir, trk_folder[0], 'graphs_info', gt_file)
            if not os.path.isfile(trk_folder_path): trk_flag = False

            if trk_flag:
                cur_trk_df = pd.read_csv(trk_folder_path, sep=' ', header=None, usecols=[0,1,3,4,5,6,7,8,9,10,11,12,13],\
                    names=['frame', 'ID', 'score', 'width', 'length', 'X', 'Y', 'Theta',\
                        'sig_X', 'sig_Y', 'sig_Theta', 'sig_l', 'sig_w'])
                # cur_trk_df = cur_trk_df[cur_trk_df.score >= 0.4]
            
            instances = cur_gt_df.ID.unique()
            for instance_ind, instance in enumerate(instances):
                print(f'- Instance #{instance_ind} -')
                save_dir = os.path.join(args.results_dir, 'diff_stats', str_dt, category, scene_name)
                os.makedirs(save_dir, exist_ok=True)
                plot_path = os.path.join(save_dir, f'{category}_{instance_ind}.png')
                csv_name = os.path.join(save_dir, f'{category}_{instance_ind}.csv')
                with open(csv_name, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

                    cur_instance_gt_df = cur_gt_df[cur_gt_df.ID == instance]
                    if len(cur_instance_gt_df) < 10: continue

                    if det_flag:
                        det_instance_df = pd.DataFrame(columns=['frame', 'score', 'width', 'length', 'X', 'Y', 'Theta'], index=range(41))
                    if trk_flag:
                        trk_instance_df = pd.DataFrame(columns=['frame', 'score', 'width', 'length', 'X', 'Y', 'Theta',\
                            'sig_X', 'sig_Y', 'sig_Theta', 'sig_l', 'sig_w'], index=range(41))
                    
                    instance_id = None
                    for _, cur_gt_row in cur_instance_gt_df.iterrows():
                        results_line = empty_line.copy()
                        results_line[:7] = [cur_gt_row['frame'], cur_gt_row['ID'],\
                            cur_gt_row['X'], cur_gt_row['Y'], cur_gt_row['Theta'], cur_gt_row['length'], cur_gt_row['width']]

                        frame_ind = cur_gt_row.frame
                        cur_gt = cur_gt_row[['width', 'length', 'X', 'Y', 'Theta']]

                        # Detection
                        if det_flag:
                            relevant_det_df = cur_det_df[cur_det_df.frame == frame_ind]
                            if not relevant_det_df.empty:
                                cur_det = relevant_det_df[['width', 'length', 'X', 'Y', 'Theta']]

                                diff = -cur_det.subtract(cur_gt, axis=1)
                                diff['Theta'] = np.where(diff['Theta'].abs() > np.pi , 2*np.pi-diff['Theta'].abs(), diff['Theta'])
                                diff_norm = diff.pow(2).sum(1).pow(0.5)
                                min_value = diff_norm.min()
                                if min_value < 8:
                                    cur_det_row = relevant_det_df.loc[diff_norm.idxmin()]
                                    results_line[7:13] = [cur_det_row['X'], cur_det_row['Y'], cur_det_row['Theta'],\
                                        cur_det_row['length'], cur_det_row['width'], cur_det_row['score']]
                                    det_instance_df.loc[frame_ind] = cur_det_row.copy()

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
                                    if min_value < 5:
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

                plot_diff_graphs(cur_instance_gt_df, det_instance_df, trk_instance_df, plot_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
