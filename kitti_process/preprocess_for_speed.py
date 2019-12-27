import os
import numpy as np
from PIL import Image
import json
import time
import numba
from numba import njit

H_MIN = -3
H_MAX = 1

D_MIN = 0
D_MAX = 80

HEIGHT = 32
WIDTH = 128

d_range = D_MAX - D_MIN
h_range = H_MAX - H_MIN

DATA_DIR = '/data/dataset/KITTI/object/'
PRO_DIR = '/data/dataset/kitti_fvnet2/projection/'
REF_DIR = '/data/dataset/kitti_fvnet2/refinement/'

def preprocess(split_list, tag='train'):
    if tag in ['train', 'val']:
        folder = 'training'
    else:
        folder = 'testing'
    data_folder = DATA_DIR + folder
    lines = open(split_list).read().split('\n')[:-1]
    start = time.time()
    for line in lines:
        # crop pointcloud
        pc_path = os.path.join(data_folder, 'velodyne', line + '.bin')
        calib_path = os.path.join(data_folder, 'calib', line + '.txt')

        pts = load_velodyne_points(pc_path)
        pts = filter_height(pts)
        P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_path)
        idx = crop_camera(pts, P, Tr_velo_to_cam, R_cam_to_rect)
        pts = pts[idx]

        front_map = get_map_and_boxes2d(pts)
        front_map_png = Image.fromarray(front_map)
        front_map_png = front_map_png.resize((512, 128), Image.NEAREST)
 
    end = time.time()
    return end-start
    

@njit(nogil=True, cache=True)
def crop_camera(pts, P, Tr_velo_to_cam, R_cam_to_rect):
    pts3d = pts.copy()
    pts3d[:, 3] = 1
    pts3d = pts3d.transpose()

    pts3d_cam = R_cam_to_rect.dot(Tr_velo_to_cam.dot(pts3d))
    idx1 = (pts3d_cam[2, :] >= 0)
    pts2d_cam = P.dot(pts3d_cam[:, idx1])
    pts1 = pts[idx1]

    pts2d_normed = pts2d_cam / pts2d_cam[2, :]

    idx2 = np.logical_and(np.logical_and(pts2d_normed[0, :] >= 0, pts2d_normed[0, :] <= 1242),
                          np.logical_and(pts2d_normed[1, :] >= 0, pts2d_normed[1, :] <= 375))
    pts2 = pts1[idx2]

    x2 = pts2[:, 0]
    y2 = pts2[:, 1]
    z2 = pts2[:, 2]

    space_dist2 = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
    plane_dist2 = np.sqrt(x2 * x2 + y2 * y2)
    azimuth2 = np.arcsin(y2 / plane_dist2)
    zenith2 = np.arcsin(z2 / space_dist2)

    A_MIN, A_MAX = azimuth2.min(), azimuth2.max()
    Z_MIN, Z_MAX = zenith2.min(), zenith2.max()

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    space_dist = np.sqrt(x * x + y * y + z * z)
    plane_dist = np.sqrt(x * x + y * y)
    azimuth = np.arcsin(y / plane_dist)
    zenith = np.arcsin(z / space_dist)

    idx = np.logical_and(np.logical_and(azimuth >= A_MIN, azimuth <= A_MAX),
                          np.logical_and(zenith >= Z_MIN, zenith <= Z_MAX))
    return idx

def fz(a):
    return a[::-1]

def FZ(mat):
    return np.array(fz(list(map(fz, mat))))

@njit(nogil=True, cache=True)
def loop(num_point, index_h, index_w, height_ratio, dist_ratio, intensity):
    front_map = np.zeros((HEIGHT, WIDTH, 3))
    for i in range(num_point):
        if (index_h[i] < HEIGHT) and (index_w[i] < WIDTH):
            dr = front_map[index_h[i], index_w[i], 1]
            if (dr == 0) or (dist_ratio[i] < dr):
                front_map[index_h[i], index_w[i], 2] = height_ratio[i]  # height ratio 0-1
                front_map[index_h[i], index_w[i], 1] = dist_ratio[i]  # dist ratio 0-1
                front_map[index_h[i], index_w[i], 0] = intensity[i]  # intensity 0-1 
    return front_map
                
def get_map_and_boxes2d(points):

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    space_dist = np.sqrt(x * x + y * y + z * z)
    plane_dist = np.sqrt(x * x + y * y)
    azimuth = np.arcsin(y / plane_dist)
    zenith = np.arcsin(z / space_dist)

    A_MIN, A_MAX = azimuth.min(), azimuth.max()
    Z_MIN, Z_MAX = zenith.min(), zenith.max()
    a_range = A_MAX - A_MIN
    z_range = Z_MAX - Z_MIN
    
    height_ratio = (z - H_MIN) / h_range
    dist_ratio = (plane_dist - D_MIN) / d_range

    index_h = np.floor((zenith - Z_MIN) / z_range * HEIGHT).astype(np.int)
    index_w = np.floor((azimuth - A_MIN) / a_range * WIDTH).astype(np.int)
    
    front_map = loop(len(points), index_h, index_w, height_ratio, dist_ratio, intensity)
    front_map = FZ(front_map)
    front_map = (front_map * 255).astype(np.uint8)
    
    return front_map

def load_velodyne_points(pc_path):
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    return points


def load_calib(calib_path):
    CAM = 2
    lines = open(calib_path).readlines()
    lines = [ list(map(lambda x:float(x), line.split()[1:])) for line in lines ][:-1]
    #P2 3*4
    P = np.array(lines[CAM]).reshape(3, 4)
    #Tr_velo_to_cam 3*4 --> 4*4
    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    #R0_rect 3*3
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)

    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect

@njit(nogil=True, cache=True)
def camera_to_lidar(points, T_VELO_2_CAM, R_RECT_0):
    N = points.shape[0]
    points_ext = np.hstack([points, np.ones((N, 1))]).T
    points_ext = np.matmul(np.linalg.inv(R_RECT_0), points_ext)
    points_ext = np.matmul(np.linalg.inv(T_VELO_2_CAM), points_ext).T
    points_ext = points_ext[:, 0:3]
    return points_ext

@njit(nogil=True, cache=True)
def filter_height(points):
    idx = np.logical_and(np.logical_and(points[:, 2] >= H_MIN, points[:, 2] <= H_MAX),
                         np.logical_and(points[:, 0] >= D_MIN, points[:, 0] <= D_MAX))
    return points[idx]
    
if __name__ == '__main__':
    t1 = preprocess('image_sets/train.txt', tag='train')
    t2 = preprocess('image_sets/val.txt', tag='val')
    t3 = preprocess('image_sets/test.txt', tag='test')
    print("used time:%.5fs, average time:%.5fs"%(t1+t2+t3, (t1+t2+t3)/14999))
