#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import ujson
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors


class vodDataset(Dataset):

    def __init__(self, args, root='/home/zyw/preprocess_res/flow_smp/', partition='train', textio=None):

        self.npoints_per_stride = 60
        self.num_strides = 10
        self.npoints = args.num_points
        self.textio = textio
        self.calib_path = 'dataset/vod_radar_calib.txt'
        self.res = {'r_res': 0.2, # m
                    'theta_res': 1.5 * np.pi/180, # radian
                    'phi_res': 1.5 *np.pi/180  # radian
                }
        self.read_calib_files()
        self.eval = args.eval
        self.partition = partition
        self.root = os.path.join(root, self.partition)
        self.interval = 0.10
        self.clips = sorted(os.listdir(self.root),key=lambda x:int(x.split("_")[1]))
        self.samples = []
        self.clips_info = []

        for clip in self.clips:
            clip_path = os.path.join(self.root, clip)
            samples = sorted(os.listdir(clip_path),key=lambda x:int(x.split("/")[-1].split("_")[0]))
            for idx in range(len(samples)):
                samples[idx] = os.path.join(clip_path, samples[idx])
            if self.eval:
                self.clips_info.append({'clip_name':clip,
                                    'index': [len(self.samples),len(self.samples)+len(samples)]
                                })
            if clip[:5] == 'delft':
                self.samples.extend(samples)

        self.textio.cprint(self.partition + ' : ' +  str(len(self.samples)))
    

    def __getitem__(self, index):
        
        sample = self.samples[index]
        with open(sample, 'rb') as fp:
            data = ujson.load(fp)

        data_1 = np.array(data["pc1"]).astype('float32')
        data_2 = np.array(data["pc2"]).astype('float32')

        # def density_estimation(data, k=10):
        #     nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(data[:, :3])
        #     distances, indices = nbrs.kneighbors(data[:, :3])
        #     densities = 1.0 / (np.mean(distances, axis=1) + 1e-6)
        #     return densities
        #
        # # 去除 ghost points
        # def remove_ghost_points(data, k=10, scale_low=0.1, scale_high=0.3):
        #     densities = density_estimation(data, k=k)
        #     threshold = np.zeros_like(densities)
        #     mask_low = densities < np.percentile(densities, 50)
        #     mask_high = densities >= np.percentile(densities, 50)
        #     threshold[mask_low] = scale_low * np.median(densities)
        #     threshold[mask_high] = scale_high * np.median(densities)
        #     mask = densities >= threshold
        #     return data[mask]
        #
        # # 去除 ghost points
        # data_1 = remove_ghost_points(data_1, k=10, scale_low=0.2, scale_high=1.0)
        # data_2 = remove_ghost_points(data_2, k=10, scale_low=0.2, scale_high=1.0)
        # # 读取data_1和data_2的第一列数据
        # data_1_col1 = data_1[:, 0]
        # data_2_col1 = data_2[:, 0]
        #
        # # 计算每两行之间的差值
        # diff_1 = np.abs(data_1_col1.reshape(-1, 1) - data_1_col1)
        # diff_2 = np.abs(data_2_col1.reshape(-1, 1) - data_2_col1)
        #
        # # 根据差值大小判断是否保留
        # mask_1 = diff_1 > 0.2
        # mask_2 = diff_2 > 0.2
        # keep_1 = np.zeros(data_1.shape[0], dtype=bool)
        # keep_2 = np.zeros(data_2.shape[0], dtype=bool)
        # for i in range(data_1.shape[0]):
        #     if np.sum(mask_1[i]) > 0:
        #         keep_1[i] = True
        # for i in range(data_2.shape[0]):
        #     if np.sum(mask_2[i]) > 0:
        #         keep_2[i] = True
        #     if not keep_2[i]:
        #         for j in range(data_2.shape[0]):
        #             if diff_2[i, j] > 0.4:
        #                 keep_2[j] = True
        #                 break
        #
        # # 保留符合条件的行
        # data_1 = data_1[keep_1]
        # data_2 = data_2[keep_2]
        # # read input data and features
        interval = self.interval
        pos_1 = data_1[:,0:3]
        pos_2 = data_2[:,0:3]

        feature_1 = data_1[:,[4,4,4]]
        feature_2 = data_2[:,[4,4,4]]

        # GT labels and pseudo FG labels (from lidar)
        gt_labels = np.array(data["gt_labels"]).astype('float32')
        pse_labels = np.array(data["pse_labels"]).astype('float32')

        # GT mask or pseudo FG mask (from lidar)
        gt_mask = np.array(data["gt_mask"])
        pse_mask = np.array(data["pse_mask"])

        # use GT labels and motion seg. mask for evaluation on val and test set
        if self.partition in ['test','val', 'train_anno']:
            labels = gt_labels
            mask = gt_mask
            opt_flow =  np.zeros((pos_1.shape[0],2)).astype('float32')
            radar_u =  np.zeros(pos_1.shape[0]).astype('float32')
            radar_v =  np.zeros(pos_1.shape[0]).astype('float32')
        # use pseudo FG flow labels and FG mask as supervision signals for training 
        else:
            labels = pse_labels
            mask = pse_mask
            opt_info = data["opt_info"]
            opt_flow = np.array(opt_info["opt_flow"]).astype('float32')
            radar_u = np.array(opt_info["radar_u"]).astype('float32')
            radar_v = np.array(opt_info["radar_v"]).astype('float32')

        # static points transformation from frame 1 to frame 2  
        trans = np.linalg.inv(np.array(data["trans"])).astype('float32')

        ## downsample to npoints to enable fast batch processing (not in test)
        if not self.eval:

            npts_1 = pos_1.shape[0]
            npts_2 = pos_2.shape[0]

            # num_strides = self.num_strides
            # sample_idx1 = []
            # for stride_idx in range(num_strides):
            #     start_idx = int(stride_idx * (npts_1 // num_strides))
            #     end_idx = int((stride_idx + 1) * (npts_1 // num_strides)) if stride_idx != num_strides - 1 else npts_1
            #     stride_points = pos_1[start_idx:end_idx]
            #     stride_npts = stride_points.shape[0]
            #     if stride_npts < self.npoints_per_stride:
            #         stride_sample_idx = np.arange(start_idx, end_idx)
            #         stride_sample_idx = np.append(stride_sample_idx, np.random.choice(stride_sample_idx,
            #                                                                           self.npoints_per_stride - stride_npts,
            #                                                                           replace=True))
            #     else:
            #         stride_sample_idx = np.linspace(start_idx, end_idx - 1, self.npoints_per_stride).astype(np.int64)
            #     sample_idx1.append(stride_sample_idx)
            #
            # sample_idx1 = np.concatenate(sample_idx1)
            #
            # # 对于data_2也做相同的修改
            # sample_idx2 = []
            # for stride_idx in range(num_strides):
            #     start_idx = int(stride_idx * (npts_2 // num_strides))
            #     end_idx = int((stride_idx + 1) * (npts_2 // num_strides)) if stride_idx != num_strides - 1 else npts_2
            #     stride_points = pos_2[start_idx:end_idx]
            #     stride_npts = stride_points.shape[0]
            #     if stride_npts < self.npoints_per_stride:
            #         stride_sample_idx = np.arange(start_idx, end_idx)
            #         stride_sample_idx = np.append(stride_sample_idx, np.random.choice(stride_sample_idx,
            #                                                                           self.npoints_per_stride - stride_npts,
            #                                                                           replace=True))
            #     else:
            #         stride_sample_idx = np.linspace(start_idx, end_idx - 1, self.npoints_per_stride).astype(np.int64)
            #     sample_idx2.append(stride_sample_idx)
            #
            # sample_idx2 = np.concatenate(sample_idx2)
            if npts_1<self.npoints:
                sample_idx1 = np.arange(0,npts_1)
                sample_idx1 = np.append(sample_idx1, np.random.choice(npts_1,self.npoints-npts_1,replace=True))
            else:
                sample_idx1 = np.random.choice(npts_1, self.npoints, replace=False)
            if npts_2<self.npoints:
                sample_idx2 = np.arange(0,npts_2)
                sample_idx2 = np.append(sample_idx2, np.random.choice(npts_2,self.npoints-npts_2,replace=True))
            else:
                sample_idx2 = np.random.choice(npts_2, self.npoints, replace=False)
            pos_1 = pos_1[sample_idx1,:]
            pos_2 = pos_2[sample_idx2,:]
            feature_1 = feature_1[sample_idx1, :]
            feature_2 = feature_2[sample_idx2, :]
            radar_u = radar_u[sample_idx1]
            radar_v = radar_v[sample_idx1]
            opt_flow = opt_flow[sample_idx1,:]

            labels = labels[sample_idx1,:]
            mask = mask[sample_idx1]

        return pos_1, pos_2, feature_1, feature_2, trans, labels, mask, interval, radar_u, radar_v, opt_flow


    def read_calib_files(self):
        with open(self.calib_path, "r") as f:
            lines = f.readlines()
            intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
            extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
            extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)
        self.camera_projection_matrix = intrinsic
        self.t_camera_radar = extrinsic

    def __len__(self):
        return len(self.samples)
