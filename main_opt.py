import os
import gc
from turtle import update
import cv2
import scipy.io
import yaml
import torch
import argparse
import numpy as np
from scipy.io import loadmat
import json
from preprocess.utils.vod.frame import homogeneous_transformation, project_3d_to_2d
from preprocess.utils.vod.visualization.settings import *
from preprocess.utils.global_param import *
from preprocess.utils.RAFT.core.raft import RAFT
from preprocess.utils.RAFT.core.utils.flow_viz import flow_to_image
from preprocess.utils.optical_flow import *


folder_path = '/home/zyw/odobeyondvision/2019-10-24-17-51-58/mmwave_middle_pcl/'
folder_path1 = '/home/zyw/odobeyondvision/2019-10-24-17-51-58/rgb/'
folder_path2 = '/home/zyw/preprocess_res/flow_smp/train/delft_2/'

folder_files = sorted(os.listdir(folder_path))
folder1_files = sorted(os.listdir(folder_path1))
folder2_files = sorted(os.listdir(folder_path2))


for i in range(0, len(folder1_files), 2):
    mat_file1_path = os.path.join(folder_path, folder_files[i])
    mat_file2_path = os.path.join(folder_path, folder_files[i+1])
    print("Reading MAT files:", mat_file1_path, mat_file2_path)
    data1 = scipy.io.loadmat(mat_file1_path)
    data2 = scipy.io.loadmat(mat_file2_path)

    jpg_file1_path = os.path.join(folder_path1, folder1_files[i])
    jpg_file2_path = os.path.join(folder_path1, folder1_files[i+1])
    print("Reading JPG files:", jpg_file1_path, jpg_file2_path)
    image1 = cv2.imread(jpg_file1_path)
    image2 = cv2.imread(jpg_file2_path)

    image1 = cv2.resize(image1, (1936, 1216))
    image2 = cv2.resize(image2, (1936, 1216))
    raft_args = argparse.Namespace(model="preprocess/utils/RAFT/raft-small.pth", \
                                   small=True, mixed_precision=False, alternate_corr=False)
    raft = RAFT(raft_args).cuda()
    raft = torch.nn.DataParallel(raft)
    raft.load_state_dict(torch.load(raft_args.model))
    opt = estimate_optical_flow(image1, image2, model=raft)
    # mat = scipy.io.loadmat('/home/zyw/odobeyondvision/2019-10-24-17-51-58/mmwave_middle_pcl/1571935920413999420.mat')
    frame = data1['frame']
    frame1 = data2['frame']
    zeros = np.zeros((frame.shape[0], 1))  # 创建一个全为0的列向量
    result = np.hstack((frame[:, :3], zeros, frame[:, 3:].reshape(-1, 1)))
    # zeros1 = np.zeros((frame.shape[0], 2))
    # new_matrix = np.hstack((result, zeros1))
    result = {"pc1": result.tolist()}

    zeros1 = np.zeros((frame1.shape[0], 1))  # 创建一个全为0的列向量
    result1 = np.hstack((frame1[:, :3], zeros1, frame1[:, 3:].reshape(-1, 1)))
    result1 = {"pc2": result1.tolist()}
    result.update(result1)

    transform_matrix = np.random.rand(4, 4)

    # 将最后一行设置为 [0, 0, 0, 1]
    transform_matrix[3] = [0, 0, 0, 1]
    result2 = {"trans": transform_matrix.tolist()}
    result.update(result2)

    radar_data1 = data1['frame'][:, :3]
    radar_p = np.concatenate((radar_data1[:, 0:3], np.ones((radar_data1.shape[0], 1))), axis=1)
    transforms1 = [1.0, 0.0, 0.0, 0.032, 0.0, 1.0, 0.0, 0.044, 0.0, 0.0, 1.0, -0.106, 0.0, 0.0, 0.0, 1.0]
    transforms1_matrix = np.array(transforms1).reshape((4, 4))
    radar_data_t = homogeneous_transformation(radar_p, transforms1_matrix)
    camera_projection = [0.468642, 0.0, 561.272442, 0.0, 0.0, 0.468642, 524.89592, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0]
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
    opt_info["radar_u"] = opt_info["radar_u"].tolist()
    opt_info["radar_v"] = opt_info["radar_v"].tolist()
    opt_info["opt_flow"] = opt_info["opt_flow"].tolist()
    result["opt_info"] = opt_info

    zeros3 = {"gt_mask": [0.0]*frame.shape[0]}
    result.update(zeros3)

    data = {"gt_labels": [[0.0, 0.0, 0.0] for _ in range(frame.shape[0])]}
    result.update(data)

    zeros4 = {"pse_mask": [1.0]*frame.shape[0]}
    result.update(zeros4)

    data3 = {"pse_labels": [[0.0, 0.0, 0.0] for _ in range(frame.shape[0])]}
    result.update(data3)
    print(result)

    json_file_path = os.path.join(folder_path2, folder2_files[i//2 % len(folder2_files)])
    print("Reading JSON file:", json_file_path)
    with open(json_file_path, "r") as f:
        data4 = json.load(f)
        data4["pc1"] = result["pc1"]
        data4["pc2"] = result["pc2"]
        data4["trans"] = result["trans"]
        data4["opt_info"] = result["opt_info"]
        data4["gt_mask"] = result["gt_mask"]
        data4["gt_labels"] = result["gt_labels"]
        data4["pse_mask"] = result["pse_mask"]
        data4["pse_labels"] = result["pse_labels"]
        with open(json_file_path, "w") as f:
            json.dump(data4, f)
print("Process finished.")

# for i in range(0, len(file_list1), 2):
#     file1_path = os.path.join(folder_path1, file_list1[i])
#     file2_path = os.path.join(folder_path1, file_list1[i+1])
#     data1 = loadmat(file1_path)
#     data2 = loadmat(file2_path)
#
# # folder_path = '/home/zyw/odobeyondvision/2019-10-24-17-51-58/rgb/'  # 文件夹路径
# # file_list = os.listdir(folder_path)  # 获取文件夹中的所有文件名
# for i in range(0, len(file_list), 2):  # 每次读取两个文件
#     file1_path = os.path.join(folder_path, file_list[i])
#     file2_path = os.path.join(folder_path, file_list[i+1])
#     image1 = cv2.imread(file1_path)
#     image2 = cv2.imread(file2_path)
# # image1 = cv2.imread("/home/zyw/odobeyondvision/2019-10-24-17-51-58/rgb/1571935919778284263.png")
# # image2 = cv2.imread("/home/zyw/odobeyondvision/2019-10-24-17-51-58/rgb/1571935919811618836.png")
# # 调整图像大小
# image1 = cv2.resize(image1, (1936, 1216))
# image2 = cv2.resize(image2, (1936, 1216))
# raft_args = argparse.Namespace(model="preprocess/utils/RAFT/raft-small.pth", \
#                                small=True, mixed_precision=False, alternate_corr=False)
# raft = RAFT(raft_args).cuda()
# raft = torch.nn.DataParallel(raft)
# raft.load_state_dict(torch.load(raft_args.model))
# opt = estimate_optical_flow(image1, image2, model=raft)
# # mat = scipy.io.loadmat('/home/zyw/odobeyondvision/2019-10-24-17-51-58/mmwave_middle_pcl/1571935920413999420.mat')
# frame = data1['frame']
# frame1 = data2['frame']
# zeros = np.zeros((frame.shape[0], 1))  # 创建一个全为0的列向量
# result = np.hstack((frame[:, :3], zeros, frame[:, 3:].reshape(-1, 1)))
# # zeros1 = np.zeros((frame.shape[0], 2))
# # new_matrix = np.hstack((result, zeros1))
# result = {"pc1": result.tolist()}
#
# zeros1 = np.zeros((frame1.shape[0], 1))  # 创建一个全为0的列向量
# result1 = np.hstack((frame1[:, :3], zeros1, frame1[:, 3:].reshape(-1, 1)))
# result1 = {"pc2": result1.tolist()}
# result.update(result1)
#
# matrix = np.zeros((4, 4))
# result2 = {"trans": matrix.tolist()}
# result.update(result2)
#
#
# radar_data1 = data1['frame'][:, :3]
# radar_p = np.concatenate((radar_data1[:, 0:3], np.ones((radar_data1.shape[0], 1))), axis=1)
# transforms1 = [1.0, 0.0, 0.0, 0.032, 0.0, 1.0, 0.0, 0.044, 0.0, 0.0, 1.0, -0.106, 0.0, 0.0, 0.0, 1.0]
# transforms1_matrix = np.array(transforms1).reshape((4, 4))
# radar_data_t = homogeneous_transformation(radar_p, transforms1_matrix)
# camera_projection = [2.468642, 0.0, 961.272442, 0.0, 0.0, 1.468642, 624.89592, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
# camera_projection_matrix = np.array(camera_projection).reshape((4, 4))
# uvs = project_3d_to_2d(radar_data_t, camera_projection_matrix)
# # filt_uv = np.logical_and(np.logical_and(uvs[:, 0] > 0, uvs[:, 0] <= IMG_WIDTH), \
# #                          np.logical_and(uvs[:, 1] > 0, uvs[:, 1] <= IMG_HEIGHT))
# # indices = np.argwhere(filt_uv).flatten()
# radar_opt = opt[uvs[:, 1] - 1, uvs[:, 0] - 1]
#
# opt_info = {"radar_u": uvs[:, 0],
#             "radar_v": uvs[:, 1],
#             "opt_flow": radar_opt,
#             }
# opt_info["radar_u"] = opt_info["radar_u"].tolist()
# opt_info["radar_v"] = opt_info["radar_v"].tolist()
# opt_info["opt_flow"] = opt_info["opt_flow"].tolist()
# result["opt_info"] = opt_info
#
# zeros3 = np.zeros((1, frame.shape[0]))
# result3 = {"gt_mask": zeros3.tolist()}
# result.update(result3)
#
# data = {"gt_labels": [[0.0, 0.0, 0.0] for _ in range(frame.shape[0])]}
# result.update(data)
#
#
# zeros4 = np.ones((1, frame.shape[0]))
# result4 = {"pse_mask": zeros4.tolist()}
# result.update(result4)
#
#
# data3 = {"pse_labels": [[0.0, 0.0, 0.0] for _ in range(frame.shape[0])]}
# result.update(data3)
# print(result)
#
# folder_path3 = '/home/zyw/preprocess_res/flow_smp/train/delft_2/'
# file_list2 = os.listdir(folder_path3)
# for i in range(0, len(file_list2), 1):
#     for filename in os.listdir(folder_path3):
#         file_path = os.path.join(folder_path3, filename)
#         with open(file_path, "r") as f:
#             data = json.load(f)
#             data["pc1"] = result["pc1"]
#             data["pc2"] = result["pc2"]
#             data["trans"] = result["trans"]
#             data["opt_info"] = result["opt_info"]
#             data["gt_mask"] = result["gt_mask"]
#             data["gt_labels"] = result["gt_labels"]
#             data["pse_mask"] = result["pse_mask"]
#             data["pse_labels"] = result["pse_labels"]
#         with open(file_path, "w") as f:
#             json.dump(data, f)
# folder_path4 = '/home/zyw/preprocess_res/flow_smp/train/delft_2/'
# for filename in os.listdir(folder_path4):
#     if filename.endswith(".json"):
#         file_path = os.path.join(folder_path4, filename)
#         with open('file_path', 'w') as f:
#             json.dump(data, f)
# flow_img = flow_to_image(opt, convert_to_bgr=True)
# vis_img = np.concatenate((image1, image2, flow_img), axis=0)
# path = 'checkpoints/'+"cmflow"+"/test_vis_flow/" + '2.jpg'
# cv2.imwrite(path, vis_img)
# print(opt_info)
# data_1 = np.array(opt_info["radar_u"]).astype('float32')
# data_2 = np.array(opt_info["radar_v"]).astype('float32')
# data_3 = np.array(opt_info["opt_flow"]).astype('float32')
# print(data_1.shape)
# print(data_2.shape)
# print(data_3.shape)