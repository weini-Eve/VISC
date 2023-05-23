from preprocess.utils.vod.visualization.vis_2d import *
from preprocess.utils.vod.visualization.vis_3d import *

import matplotlib.pyplot as plt
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from vod.visualization import Visualization2D
from vod.visualization import Visualization3D

kitti_locations = KittiLocations(root_dir="/home/zyw/view_of_delft/",
                                output_dir="/home/zyw/1/")

frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                             frame_number="01801")
vis2d = Visualization2D(frame_data)

vis2d.draw_plot(show_lidar=False,
                show_radar=True,
                show_gt=False,
                min_distance_threshold=5,
                max_distance_threshold=20)

# imgplot = plt.imshow(frame_data.image)
# plt.show()

# print(frame_data.lidar_data)
#
# # 3D Visualization of the point-cloud
# vis_3d = Visualization3D(frame_data=frame_data)
# #
# #
# vis_3d.draw_plot(radar_origin_plot=True,
#                    lidar_origin_plot=True,
#                    camera_origin_plot=True,
#                    lidar_points_plot=True,
#                    radar_points_plot=True,
#                    radar_velocity_plot=True,
#                    annotations_plot=True)






