from operator import inv

import ujson
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


def extract_data_info(data):

    pc1, pc2, ft1, ft2, trans, gt, mask, interval, radar_u, radar_v, opt_flow = data
    pc1 = pc1.cuda().transpose(2,1).contiguous()
    pc2 = pc2.cuda().transpose(2,1).contiguous()
    ft1 = ft1.cuda().transpose(2,1).contiguous()
    ft2 = ft2.cuda().transpose(2,1).contiguous()
    radar_v = radar_v.cuda().float()
    radar_u = radar_u.cuda().float()
    opt_flow = opt_flow.cuda().float()
    mask = mask.cuda().float()
    trans = trans.cuda().float()
    interval = interval.cuda().float()
    gt = gt.cuda().float()

    return pc1, pc2, ft1, ft2, trans, gt, mask, interval, radar_u, radar_v, opt_flow

# with open('/home/zyw/preprocess_res2/flow_smp/train/delft_2/00544_00545.json', 'r') as g:
#     data = ujson.load(g)
# data_1 = np.array(data["pc1"]).astype('float32')
#
# trans = np.array(data["trans"]).astype('float32')
# print(trans)
# mean = 0
# std_dev = 1
# noise = np.random.normal(mean, std_dev, (4, 1))
# trans[:, -1] += noise[:, 0]
# print(trans)
with open('/home/zyw/view_of_delft/radar_3frames/training/pose/00000.json', 'r') as g:
    data = ujson.load(g)
data_1 = np.array(data["odomToCamera"]).astype('float32')
