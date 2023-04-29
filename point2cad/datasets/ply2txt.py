import numpy as np
import open3d as o3d
import os
from plyfile import PlyData
data_path = 'D:/paper_code/point_transformer/Point-Transformers-master/data/pc_cad'
save_path = 'D:/paper_code/point_transformer/Point-Transformers-master/data/deepcad_txt'
if not os.path.exists(save_path):
    os.makedirs(save_path)
names = os.listdir(data_path)
names.sort()
print(names)
number = 0
for name in names:
    file_path = os.path.join(data_path,name)
    point=o3d.io.read_point_cloud(file_path)
    pointxyz=np.asarray(point.points)
    pointcolor=np.asarray(point.colors)
    #print('pointxyz: ',pointxyz)
    #print('pointcolor: ',pointcolor)
    #print('pointxyz.shape: ',pointxyz.shape)
    #print('pointcolor.shape: ',pointcolor.shape)
    pointcloud = np.concatenate((pointxyz,pointcolor), axis=0)
    #print('pointcloud.shape: ',pointcloud.shape)
    file_save_path = os.path.join(save_path,f'{name[:-4]}.txt')
    np.savetxt(file_save_path, pointcloud,fmt='%.6f')
    print('number: ',number)
    number += 1
# randomlist=np.random.choice(pointxyz.shape[0],size=(self.num))
# pointcloud[idx,:,:3]=pointxyz[randomlist]
# pointcloud[idx,:,3:]=pointcolor[randomlist]


# def read_ply_cloud(filename):
#     ply_data = PlyData.read(filename)
#     points = ply_data['vertex'].data.copy()
#     print(points.shape)
#     cloud = np.empty([6513005, 3])
#     for i in range(len(points)):
#         point = points[i]
#         p = np.array([point[0], point[1], point[2]])
#         cloud[i] = p
#     return np.array(cloud)
#
# out_arr = read_ply_cloud('D:/paper_code/point_transformer/Point-Transformers-master/data/1.ply')
#
# np.savetxt(r'D:/paper_code/point_transformer/Point-Transformers-master/data/testPLY.txt', out_arr, fmt='%d')
# print("output array from input list : ", out_arr)
# print("shape : ", out_arr.shape)
