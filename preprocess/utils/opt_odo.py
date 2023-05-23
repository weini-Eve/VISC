import ujson
from tqdm import tqdm
import matplotlib.pyplot as plt
from .common import get_frame_list
from scipy.spatial.transform import Rotation as R
from .vod.configuration import KittiLocations
from .vod.frame import FrameDataLoader
from .vod.frame import FrameTransformMatrix
from .vod.frame import homogeneous_transformation, project_3d_to_2d
from .vod.frame import FrameLabels
from .vod.visualization.helpers import get_transformed_3d_label_corners
from .vod.visualization.settings import *
from .global_param import *
from .RAFT.core.raft import RAFT
from .RAFT.core.utils.flow_viz import flow_to_image
from .optical_flow import *
import get_flow_samples

raft_model = init_raft()
img1 = cv2.cvtColor(data1.image, cv2.COLOR_RGB2BGR)
img2 = cv2.cvtColor(data2.image, cv2.COLOR_RGB2BGR)
opt_flow = estimate_optical_flow(img1, img2, raft_model)
show_optical_flow(img1, img2, opt_flow, opt_path, frame1)
opt_info = info_from_opt_flow(radar_data1, transforms1, opt_flow)