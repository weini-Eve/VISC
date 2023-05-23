import numpy as np
from matplotlib import pyplot as plt
import numpy as np
# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.load('/home/zyw/data/test/radar/8.npy')[:,:3])
# o3d.visualization.draw_geometries([pcd])
# 设置文件路径
file_path = '/home/zyw/data/test/radar/8.npy'

# 加载.npy文件
data = np.load(file_path)
fig = plt.figure(figsize=(10, 5))

# 绘制 data_1
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(data[:,0], data[:,1], data[:,2], s=1)
ax1.set_title('Data 1')
plt.show()
# 打印数据
print(data)
