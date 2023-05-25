import os
import cv2
import yaml
import torch
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.common import get_frame_list
from utils.vod.configuration import MilliegoLocations
from utils.get_flow_samples import get_radar_flow_from_milliego

TASK = 'scene_flow'


def main(args):
    root_dir = args.root_dir
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # path for saving all scene flow samples
    smp_path = os.path.join(save_dir, 'flow_smp')
    # path for saving optical flow visualization (from RAFT)
    opt_path = os.path.join(save_dir, 'opt_vis')

    sub_dirs = os.listdir(root_dir)
    for sub_dir in sub_dirs:
        data_loc = MilliegoLocations(root_dir=root_dir, sub_dir=sub_dir)
        get_radar_flow_from_milliego(data_loc, smp_path, opt_path, mode ='train')

    # for split in splits:
    #     for clip in splits[split]:
    #         frames = get_frame_list(clip_path + '/' + clip + '.txt')
    #     if frames is not None:
    #         # aggregate cross-modal info and consecutive radar pcs for training, validation and testing
    #         if split == 'train':
    #             get_radar_flow_samples(data_loc, frames, smp_path, opt_path, clip, split, pseudo_label_path,
    #                                    mode='train')
    #         if split == 'val':
    #             get_radar_flow_samples(data_loc, frames, smp_path, opt_path, clip, split, true_label_path, mode='val')
    #         if split == 'test':
    #             get_radar_flow_samples(data_loc, frames, smp_path, opt_path, clip, split, true_label_path, mode='test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--root_dir', type=str, default='/home/clarence/MilliegoDataset',
                        help='Path for the original dataset.')
    parser.add_argument('--save_dir', type=str, default='/home/clarence/radarflow/PreprocessingResults', help='Path for saving preprocessing results.')
    args = parser.parse_args()
    main(args)