import os
import gc
from turtle import update
import cv2
import yaml
import torch
import argparse
import numpy as np
from preprocess.utils.vod.frame import homogeneous_transformation, project_3d_to_2d
from preprocess.utils.vod.visualization.settings import *
from preprocess.utils.global_param import *
from preprocess.utils.RAFT.core.raft import RAFT
from preprocess.utils.RAFT.core.utils.flow_viz import flow_to_image
from scipy.optimize import minimize

def estimate_optical_flow(img1,img2,model):

    resize_dim = (int(RESIZE_SCALE*img1.shape[1]),int(RESIZE_SCALE*img1.shape[0]))
    img1 = cv2.resize(img1,resize_dim)
    img2 = cv2.resize(img2,resize_dim)
    img1_torch = torch.from_numpy(img1).cuda().unsqueeze(0).transpose(1,3)
    img2_torch = torch.from_numpy(img2).cuda().unsqueeze(0).transpose(1,3)
    opt_flow = model(img1_torch, img2_torch, 12)
    np_flow = opt_flow.squeeze(0).permute(2,1,0).cpu().detach().numpy()
    resize_dim = (int(img1.shape[1]/RESIZE_SCALE),int(img1.shape[0]/RESIZE_SCALE))
    flow = cv2.resize(np_flow, resize_dim)

    return flow


def init_raft():

    # parser = argparse.ArgumentParser()
    
    # parser.add_argument('--model', default= "preprocess/utils/RAFT/raft-small.pth", help="restore checkpoint")
    # parser.add_argument('--path', help="dataset for evaluation")
    # parser.add_argument('--small', action='store_false', help='use small model')
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    # raft_args = parser.parse_args()
    raft_args = argparse.Namespace(model = "utils/RAFT/raft-small.pth",\
                                    small = True,mixed_precision = False,alternate_corr = False)
    raft = RAFT(raft_args).cuda()
    raft = torch.nn.DataParallel(raft)
    raft.load_state_dict(torch.load(raft_args.model))

    return raft


def show_optical_flow(img1, img2, opt_flow, opt_path, frame1):

    flow_img = flow_to_image(opt_flow, convert_to_bgr=True)
    vis_img = np.concatenate((img1,img2,flow_img),axis=0)
    path = opt_path + '/' + frame1 + '.jpg'
    cv2.imwrite(path, vis_img)


def info_from_opt_flow(radar_data, transforms, opt_flow):

    radar_p = np.concatenate((radar_data[:,0:3],np.ones((radar_data.shape[0],1))),axis=1)
    radar_data_t = homogeneous_transformation(radar_p, transforms.t_camera_radar)
    uvs = project_3d_to_2d(radar_data_t,transforms.camera_projection_matrix)
    #filt_uv = np.logical_and(np.logical_and(uvs[:,0]>0, uvs[:,0]<opt_flow.shape[1]),\
    #     np.logical_and(uvs[:,1]>0, uvs[:,1]<opt_flow.shape[0]))

    radar_opt = opt_flow[uvs[:,1]-1,uvs[:,0]-1]

    opt_info = {"radar_u": uvs[:,0],
                "radar_v": uvs[:,1],
                "opt_flow": radar_opt,
                }

    return opt_info



def filt_points_in_fov(pc_data, transforms, sensor):

    pc_h = np.concatenate((pc_data[:,0:3],np.ones((pc_data.shape[0],1))),axis=1)
    if sensor == 'radar':
        pc_cam = homogeneous_transformation(pc_h, transforms.t_camera_radar)
    if sensor == 'lidar':
        pc_cam = homogeneous_transformation(pc_h, transforms.t_camera_lidar)
    uvs = project_3d_to_2d(pc_cam,transforms.camera_projection_matrix)
    filt_uv = np.logical_and(np.logical_and(uvs[:,0]>0, uvs[:,0]<=IMG_WIDTH),\
         np.logical_and(uvs[:,1]>0, uvs[:,1]<=IMG_HEIGHT))
    indices = np.argwhere(filt_uv).flatten()

    return indices


def cam_trans(img1, img2):
    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 分别提取各自的SIFT特征
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 创建FLANN匹配器
    matcher = cv2.FlannBasedMatcher()

    # 特征匹配
    matches = matcher.match(des1, des2)

    # 取前10个最佳匹配点对
    matches = sorted(matches, key=lambda x: x.distance)[:100]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 估计相机的本质矩阵和外部参数
    E_metric, mask = cv2.findEssentialMat(np.float32([kp1[m.queryIdx].pt for m in matches]),
                               np.float32([kp2[m.trainIdx].pt for m in matches]),
                               focal=1.0, pp=(0, 0))

    # 估计相机的外部参数
    K = [566.8943529201453, 0.0, 322.10094802162763, 0.0, 567.7699123433893, 242.8149724252196, 0.0, 0.0, 1.0]
    K = np.array(K).reshape((3, 3))
    _, Rot_cam, trans_cam, mask = cv2.recoverPose(E_metric, np.float32([kp1[m.queryIdx].pt for m in matches]),
                                np.float32([kp2[m.trainIdx].pt for m in matches]),
                                cameraMatrix=K, mask=mask)

    projMat1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    projMat2 = np.hstack((Rot_cam, trans_cam))

    # 三角化得到3D点坐标
    points_4d = cv2.triangulatePoints(projMat1, projMat2, pts1, pts2)

    # 将4D坐标转化为3D坐标
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    # 使用PnP算法求解相机的运动
    retval, rvec, tvec = cv2.solvePnP(points_3d, pts2, K, None)

    # 将旋转向量转化为旋转矩阵
    Rot, _ = cv2.Rodrigues(rvec)

    # 使用cv2.projectPoints函数将3D点投影到图像平面上，得到2D点的预测值
    pts2_pred, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)

    # 计算预测值和实际值之间的差距，即重投影误差
    reproj_error = np.sqrt(np.mean((pts2_pred - pts2) ** 2))
    return Rot, tvec, points_3d, K, pts2
def IMU_trans(imu_data):
    # 获取第15-17列和第20-22列的数据
    # data = IMU_data.iloc[:, [0, 14, 15, 16, 19, 20, 21]]
    # imu_data = np.array(data)
    # 加载IMU数据
    # 定义变量
    dt = 0  # 采样时间
    g = np.array([0, 0, -9.8])  # 重力加速度
    Rot_imu = np.eye(3)  # 初始旋转矩阵
    tans_imu = np.zeros(3)  # 初始位置
    v = np.zeros(3)  # 初始速度
    bias_gyro = [0.0004, 0.0, 0.0]
    bias_gyro_matrix = np.diag(bias_gyro)  # 陀螺仪偏差
    bias_acc = [0.0004, 0.0, 0.0]
    bias_acc_matrix = np.diag(bias_acc)  # 加速度计偏差

    # 预积分
    for i in range(1, imu_data.shape[0]):
        # 计算时间间隔
        dt_i = imu_data[i, 0] - imu_data[i - 1, 0]
        dt_i = dt_i * 10 ** -8
        # 计算旋转增量
        omega = (imu_data[i, 4:7] - bias_gyro_matrix.dot(imu_data[i, 1:4])) * dt_i
        Rot_imu = Rot_imu.dot(np.array([[1, -omega[2], omega[1]],
                                        [omega[2], 1, -omega[0]],
                                        [-omega[1], omega[0], 1]]))

        # 计算加速度增量
        a = (imu_data[i, 1:4] - bias_acc_matrix.dot(imu_data[i, 1:4]))
        a = Rot_imu.dot(a) - g
        v += a * dt_i
        tans_imu += v * dt_i + 0.5 * a * dt_i ** 2
        tans_imu1 = tans_imu.reshape((3, 1))
        extrinsic_imu_to_cam = np.array([[-0.99973756, -0.01087666, 0.02016219, 0.03243579],
                                        [-0.0207632, 0.05830964, -0.9980826, 0.16294065],
                                        [0.00968016, -0.99823929, -0.05852017, 0.3316231]])

        # 3x3 rotation matrix of camera coordinate system
        R_cam = extrinsic_imu_to_cam[:, :3]

        # 3x1 translation matrix of camera coordinate system
        T_cam = extrinsic_imu_to_cam[:, 3:]

        # transformation matrix from IMU to camera coordinate system
        R_imu2cam = np.dot(R_cam, Rot_imu)  # 3x3
        T_imu2cam = np.dot(R_cam, tans_imu1) + T_cam  # 3x1
    return R_imu2cam, T_imu2cam, imu_data, bias_gyro_matrix, bias_acc_matrix

def optimizeVIO(img1, img2, IMU_data):
    Rot, tvec, points_3d, K, pts2 = cam_trans(img1, img2)
    R_imu2cam, T_imu2cam, imu_data, bias_gyro_matrix, bias_acc_matrix = IMU_trans(IMU_data)
    def optimize(x):
        R_new = x[:9].reshape(3, 3)
        t_new = x[9:]
        pts2_pred, _ = cv2.projectPoints(points_3d, R_new, t_new, K, None)
        # R_new = np.reshape(x[12:21], (3, 3))
        # t_new  = x[21:24]
        error = np.zeros(6)
        # 预积分
        v = np.zeros(3)
        g = np.array([0, 0, -9.8])
        for i in range(1, imu_data.shape[0]):
            # 计算时间间隔
            dt_i = imu_data[i, 0] - imu_data[i - 1, 0]
            dt_i = dt_i * 10 ** -8
            # 计算旋转增量
            omega = (imu_data[i, 1:4] - bias_gyro_matrix.dot(imu_data[i, 1:4])) * dt_i
            R_new = R_new.dot(np.array([[1, -omega[2], omega[1]],
                                            [omega[2], 1, -omega[0]],
                                            [-omega[1], omega[0], 1]]))

            # 计算加速度增量
            a = (imu_data[i, 4:7] - bias_acc_matrix.dot(imu_data[i, 4:7]))
            a = R_new.dot(a) - g
            v += a * dt_i
            t_new += v * dt_i + 0.5 * a * dt_i ** 2
            # 计算预积分误差
            error[0:3] += (t_new - imu_data[i, 4:7]) * dt_i
            error[3:6] += (R_new.dot(imu_data[i, 1:4]) - imu_data[i, 1:4]) * dt_i
            return np.sqrt(np.mean((pts2_pred - pts2) ** 2)) + np.sum(error ** 2)

    Rot_opt = R_imu2cam + Rot
    Trans_opt = T_imu2cam + tvec

    x0 = np.concatenate((Rot_opt, Trans_opt), axis=1).reshape(-1)
    res = minimize(optimize, x0, method='BFGS')
    R_opt = np.reshape(res.x[0:9], (3, 3))
    t_opt = res.x[9:12].reshape((3, 1))
    return R_opt, t_opt