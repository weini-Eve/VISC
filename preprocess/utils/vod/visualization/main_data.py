import ujson
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
# with open('/home/zyw/preprocess_res1/flow_smp/train/delft_2/00544_00545.json', 'r') as f:
# data = ujson.load(f)
# data_1 = np.array(data["pc1"]).astype('float32')
# data_2 = np.array(data["pc2"]).astype('float32')
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].scatter(data_2[:,0], data_2[:,1], s=1)
# axs[0].set_title('Data 2')

# with open('/home/zyw/preprocess_res2/flow_smp/train/delft_2/00544_00545.json', 'r') as g:
#     data = ujson.load(g)
with open('/home/zyw/preprocess_res/flow_smp/train/delft_2/00544_00545.json', 'r') as g:
    data = ujson.load(g)
data_1 = np.array(data["pc1"]).astype('float32')
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].scatter(data_1[:,0], data_1[:,1], s=1)
# axs[0].set_title('Data 2')
data_2 = np.array(data["pc2"]).astype('float32')
data_3 = np.array(data["trans"]).astype('float32')
data_4 = np.array(data["gt_mask"]).astype('float32')
data_5 = np.array(data["gt_labels"]).astype('float32')
data_6 = np.array(data["pse_mask"]).astype('float32')
data_7 = np.array(data["pse_labels"]).astype('float32')
# axs[1].scatter(data_2[:,0], data_2[:,1], s=1)
# axs[1].set_title('Data 3')
# plt.show()
# 从点云数据中提取xyz坐标信息
# 从N*5格式的数据中提取xyz坐标

# 计算点云的密度估计
def density_estimation(data, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(data[:,:3])
    distances, indices = nbrs.kneighbors(data[:,:3])
    densities = 1.0 / (np.mean(distances, axis=1) + 1e-6)
    return densities

# 去除 ghost points
def remove_ghost_points(data, k=10, scale_low=0.1, scale_high=0.3):
    densities = density_estimation(data, k=k)
    threshold = np.zeros_like(densities)
    mask_low = densities < np.percentile(densities, 50)
    mask_high = densities >= np.percentile(densities, 50)
    threshold[mask_low] = scale_low * np.median(densities)
    threshold[mask_high] = scale_high * np.median(densities)
    mask = densities >= threshold
    return data[mask]
# 去除 ghost points

print(data_1.shape)
print(data_2.shape)
print(data_3.shape)
print(data_4.shape)
print(data_5.shape)
print(data_6.shape)
print(data_7.shape)

# # 去除 ghost points
# data_1_clean = remove_ghost_points(data_1, k=10, scale_low=0.3, scale_high=1.0)
# data_2_clean = remove_ghost_points(data_2, k=10, scale_low=0.3, scale_high=1.0)
#
# print("去除 ghost points 后，data_1 中剩余点数：", len(data_1_clean))
# print("去除 ghost points 后，data_2 中剩余点数：", len(data_2_clean))




# # 将 RRV 和 RCS 信息拼接到点云坐标后面
# data_1_concat = np.concatenate((data_1[:, :3], data_1[:, 3:]), axis=1)
# data_2_concat = np.concatenate((data_2[:, :3], data_2[:, 3:]), axis=1)
#
# # 1. KDTree 点云匹配
# tree = KDTree(data_1_concat)
# matched_distances, matched_indices = tree.query(data_2_concat)
#
# # 2. 丢弃未匹配的点并保存匹配结果
# data_3 = []
# data_4 = []
# for i, index in enumerate(matched_indices):
#     if index != -1:
#         data_3.append(data_1[index])
#         data_4.append(data_2[i])
#
# data_3 = np.array(data_3)
# data_4 = np.array(data_4)
#
# cloud = pcl.PointCloud.PointXYZRCVRGVG()
# cloud.from_array(data_3)
#
# # Create a statistical outlier removal filter
# outlier_filter = pcl.filters.StatisticalOutlierRemoval.PointXYZRCVRGVG()
#
# # Set the filter parameters
# outlier_filter.setInputCloud(cloud)
# outlier_filter.setMeanK(50)  # Number of neighbors to use for mean distance estimation
# outlier_filter.setStddevMulThresh(1.0)  # Standard deviation multiplier
#
# # Apply the filter
# filtered_cloud = pcl.PointCloud.PointXYZRCVRGVG()
# outlier_filter.filter(filtered_cloud)
#
# # Convert the filtered cloud back to a numpy array
# filtered_data = np.asarray(filtered_cloud)
#
#
# fig, axs = plt.subplots(1, 4, figsize=(10, 5))
# axs[0].scatter(data_1[:,0], data_1[:,1], s=1)
# axs[0].set_title('Data 0')
# axs[1].scatter(data_2[:,0], data_2[:,1], s=1)
# axs[1].set_title('Data 1')
# axs[2].scatter(filtered_data[:,0], filtered_data[:,1], s=1)
# axs[2].set_title('Data 2')
# axs[3].scatter(data_4[:,0], data_4[:,1], s=1)
# axs[3].set_title('Data 3')
# plt.show()
# # 读取data_1和data_2的第一列数据
# data_1_col1 = data_1[:, 0]
# data_3_col1 = data_3[:, 0]
#
# # 计算每两行之间的差值
# diff_3 = np.abs(data_3_col1.reshape(-1, 1) - data_3_col1)
#
# # 根据差值大小判断是否保留
# mask_3 = diff_3 < 10
# keep_3 = np.zeros(data_3.shape[0], dtype=bool)
# for i in range(data_3.shape[0]):
#     if np.sum(mask_3[i]) > 0:
#         keep_3[i] = True
#     if not keep_3[i]:
#         for j in range(data_3.shape[0]):
#             if diff_3[i, j] < 10:
#                 keep_3[j] = True
#                 break
#
# # 保留符合条件的行
# data_4 = data_3[keep_3]
# axs[2].scatter(data_4[:,0], data_4[:,1], s=1)
# axs[2].set_title('Data 4')


# pos_1 = data_1[:, 0:3]
# pos_2 = data_2[:, 0:3]
# npts_1 = pos_1.shape[0]
# npts_2 = pos_2.shape[0]
#
# num_strides =20
# npoints_per_stride = 50
# num_strides = num_strides
# sample_idx1 = []
#
# for stride_idx in range(num_strides):
#     start_idx = int(stride_idx * (npts_1 // num_strides))
#     end_idx = int((stride_idx + 1) * (npts_1 // num_strides)) if stride_idx != num_strides - 1 else npts_1
#     stride_points = pos_1[start_idx:end_idx]
#     stride_npts = stride_points.shape[0]
#     if stride_npts < npoints_per_stride:
#         stride_sample_idx = np.arange(start_idx, end_idx)
#         stride_sample_idx = np.append(stride_sample_idx, np.random.choice(stride_sample_idx,
#                                                                           npoints_per_stride - stride_npts,
#                                                                           replace=True))
#     else:
#         stride_sample_idx = np.linspace(start_idx, end_idx - 1, npoints_per_stride).astype(np.int64)
#     sample_idx1.append(stride_sample_idx)
#
# sample_idx1 = np.concatenate(sample_idx1)
# pos_1 = pos_1[sample_idx1, :]
#
# fig = plt.figure(figsize=(10, 10))
# ax1 = fig.add_subplot(223, projection='3d')
# ax2 = fig.add_subplot(224, projection='3d')
#
# # 绘制 data_1
# ax3 = fig.add_subplot(2, 2, 1, projection='3d')
# ax3.scatter(data_1[:,0], data_1[:,1], data_1[:,2], s=1)
# ax3.set_title('Data 1')
#
# # 绘制 data_2
# ax4 = fig.add_subplot(2, 2, 2, projection='3d')
# ax4.scatter(data_2[:,0], data_2[:,1], data_2[:,2], s=1)
# ax4.set_title('Data 2')
#
# for stride_idx in range(num_strides):
#     start_idx = int(stride_idx * (npoints_per_stride))
#     end_idx = int((stride_idx + 1) * (npoints_per_stride)) if stride_idx != num_strides - 1 else pos_1.shape[0]
#     stride_points = pos_1[start_idx:end_idx]
#     ax1.scatter(stride_points[:, 0], stride_points[:, 1], stride_points[:, 2], marker='s', s=50)
#
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.set_title('Pose 1')
# # 对于data_2也做相同的修改
# sample_idx2 = []
# for stride_idx in range(num_strides):
#     start_idx = int(stride_idx * (npts_2 // num_strides))
#     end_idx = int((stride_idx + 1) * (npts_2 // num_strides)) if stride_idx != num_strides - 1 else npts_2
#     stride_points = pos_2[start_idx:end_idx]
#     stride_npts = stride_points.shape[0]
#     if stride_npts < npoints_per_stride:
#         stride_sample_idx = np.arange(start_idx, end_idx)
#         stride_sample_idx = np.append(stride_sample_idx, np.random.choice(stride_sample_idx,
#                                                                           npoints_per_stride - stride_npts,
#                                                                           replace=True))
#     else:
#         stride_sample_idx = np.linspace(start_idx, end_idx - 1, npoints_per_stride).astype(np.int64)
#     sample_idx2.append(stride_sample_idx)
#
# sample_idx2 = np.concatenate(sample_idx2)
# pos_2 = pos_2[sample_idx2, :]
#
# for stride_idx in range(num_strides):
#     start_idx = int(stride_idx * (npoints_per_stride))
#     end_idx = int((stride_idx + 1) * (npoints_per_stride)) if stride_idx != num_strides - 1 else pos_2.shape[0]
#     stride_points = pos_2[start_idx:end_idx]
#     ax2.scatter(stride_points[:, 0], stride_points[:, 1], stride_points[:, 2], marker='s', s=50)
#
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.set_title('Pose 2')
#
# plt.show()
# if npts_1 < 256:
#     sample_idx1 = np.arange(0, npts_1)
#     sample_idx1 = np.append(sample_idx1, np.random.choice(npts_1, 256 - npts_1, replace=True))
# else:
#     sample_idx1 = np.random.choice(npts_1, 256, replace=False)
# if npts_2 < 256:
#     sample_idx2 = np.arange(0, npts_2)
#     sample_idx2 = np.append(sample_idx2, np.random.choice(npts_2, 256 - npts_2, replace=True))
# else:
#     sample_idx2 = np.random.choice(npts_2, 256, replace=False)
# pos_1 = pos_1[sample_idx1, :]
# pos_2 = pos_2[sample_idx2, :]
# print(pos_1.shape)
# print(pos_2.shape)
fig = plt.figure(figsize=(10, 5))

# 绘制 data_1
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(data_1[:,0], data_1[:,1], data_1[:,2], s=1)
ax1.set_title('Data 1')

# 绘制 data_2
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(data_2[:,0], data_2[:,1], data_2[:,2], s=1)
ax2.set_title('Data 2')
#
# ax2 = fig.add_subplot(2, 3, 3, projection='3d')
# ax2.scatter(data_1_clean[:,0], data_1_clean[:,1], data_1_clean[:,2], s=1)
# ax2.set_title('pos_1')
#
# ax2 = fig.add_subplot(2, 3, 4, projection='3d')
# ax2.scatter(data_2_clean[:,0], data_2_clean[:,1], data_2_clean[:,2], s=1)
# ax2.set_title('pos_1')
# # 绘制 data_2
# ax2 = fig.add_subplot(2, 3, 5, projection='3d')
# ax2.scatter(pos_1[:,0], pos_1[:,1], pos_1[:,2], s=1)
# ax2.set_title('pos_1')
#
# # 绘制 data_2
# ax2 = fig.add_subplot(2, 3, 6, projection='3d')
# ax2.scatter(pos_2[:,0], pos_2[:,1], pos_2[:,2], s=1)
# ax2.set_title('pos_2')
#
plt.show()

