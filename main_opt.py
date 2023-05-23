import os
import gc
from turtle import update
import cv2
import scipy.io
import yaml
import torch
import argparse
import numpy as np
from preprocess.utils.vod.frame import homogeneous_transformation, project_3d_to_2d
from preprocess.utils.vod.visualization.settings import *
from preprocess.utils.global_param import *
from preprocess.utils.RAFT.core.raft import RAFT
from preprocess.utils.RAFT.core.utils.flow_viz import flow_to_image
from preprocess.utils.optical_flow import *

image1 = cv2.imread("/home/zyw/odobeyondvision/2019-10-24-17-51-58/rgb/1571935919106706885.png")
image2 = cv2.imread("/home/zyw/odobeyondvision/2019-10-24-17-51-58/rgb/1571935919140264279.png")
# 调整图像大小
image1 = cv2.resize(image1, (1936, 1216))
image2 = cv2.resize(image2, (1936, 1216))
raft_args = argparse.Namespace(model="preprocess/utils/RAFT/raft-small.pth", \
                               small=True, mixed_precision=False, alternate_corr=False)
raft = RAFT(raft_args).cuda()
raft = torch.nn.DataParallel(raft)
raft.load_state_dict(torch.load(raft_args.model))
opt = estimate_optical_flow(image1, image2, model=raft)
mat = scipy.io.loadmat('/home/zyw/odobeyondvision/2019-10-24-17-51-58/mmwave_middle_pcl/1571935919364126164.mat')
frame = mat['frame']
radar_data1 = mat['frame'][:, :3]
radar_p = np.concatenate((radar_data1[:, 0:3], np.ones((radar_data1.shape[0], 1))), axis=1)
transforms1 = [1.0, 0.0, 0.0, 0.032, 0.0, 1.0, 0.0, 0.044, 0.0, 0.0, 1.0, -0.106, 0.0, 0.0, 0.0, 1.0]
transforms1_matrix = np.array(transforms1).reshape((4, 4))
radar_data_t = homogeneous_transformation(radar_p, transforms1_matrix)
camera_projection = [3.468642, 0.0, 961.272442, 0.0, 0.0, 3.468642, 624.89592, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
camera_projection_matrix = np.array(camera_projection).reshape((4, 4))
uvs = project_3d_to_2d(radar_data_t, camera_projection_matrix)
# filt_uv = np.logical_and(np.logical_and(uvs[:, 0] > 0, uvs[:, 0] <= IMG_WIDTH), \
#                          np.logical_and(uvs[:, 1] > 0, uvs[:, 1] <= IMG_HEIGHT))
# indices = np.argwhere(filt_uv).flatten()
radar_opt = opt[uvs[:, 1] - 1, uvs[:, 0] - 1]

opt_info = {"radar_u": uvs[:, 0],
            "radar_v": uvs[:, 1],
            "opt_flow": radar_opt,
            }
flow_img = flow_to_image(opt, convert_to_bgr=True)
vis_img = np.concatenate((image1, image2, flow_img), axis=0)
path = 'checkpoints/'+"cmflow"+"/test_vis_flow/" + '2.jpg'
cv2.imwrite(path, vis_img)
print(opt_info)