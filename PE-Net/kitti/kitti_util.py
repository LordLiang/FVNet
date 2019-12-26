import os
import numpy as np
from box_util import box3d_iou
DIST_RANGE = 80
#lwh
g_mean_size_arr = np.array([3.88311640418,1.62856739989,1.52563191462])

NUM_ANGLE_BIN = 6


def crop_points(pc_velo, front_bbox):

    ar1, ar2, zr1, zr2, dr1, dr2 = front_bbox
    x = pc_velo[:, 0]
    y = pc_velo[:, 1]
    z = pc_velo[:, 2]

    space_dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    plane_dist = np.sqrt(x ** 2 + y ** 2)
    azimuth = np.arcsin(y / plane_dist)
    zenith = np.arcsin(z / space_dist)

    A_MIN, A_MAX = azimuth.min(), azimuth.max()
    Z_MIN, Z_MAX = zenith.min(), zenith.max()
    a_range = A_MAX - A_MIN
    z_range = Z_MAX - Z_MIN

    a1 = ar1 * a_range + A_MIN
    a2 = ar2 * a_range + A_MIN
    z1 = zr1 * z_range + Z_MIN
    z2 = zr2 * z_range + Z_MIN
    d1 = dr1 * DIST_RANGE
    d2 = dr2 * DIST_RANGE

    idx = np.logical_and(np.logical_and(np.logical_and((azimuth >= a1), (azimuth <= a2)),
                                        np.logical_and((zenith >= z1), (zenith <= z2))),
                         np.logical_and((plane_dist >= d1), (plane_dist <= d2)))
    cropped_obj = pc_velo[idx]

    return cropped_obj


def random_perturb_box(bbox):
    x1, x2, y1, y2, r1, r2 = bbox
    w = x2 - x1
    h = y2 - y1
    dist = r2 - r1
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    cd = r1 + dist/2.0
    cx_new = cx + np.random.uniform(low=-0.1*w, high=0.1*w)
    cy_new = cy + np.random.uniform(low=-0.1*h, high=0.1*h)
    cd_new = cd + np.random.uniform(low=-0.1*dist, high=0.1*dist)
    w_new = w * np.random.uniform(low=0.9, high=1.1)
    h_new = h * np.random.uniform(low=0.9, high=1.1)
    dist_new = dist * np.random.uniform(low=0.9, high=1.1)
    bbox_new = np.array([cx_new - w_new / 2.0, cx_new + w_new / 2.0,
                         cy_new - h_new / 2.0, cy_new + h_new / 2.0,
                         cd_new - dist_new / 2.0, cd_new + dist_new / 2.0])
    return bbox_new


def lidar_to_camera_point(pc_velo, T_VELO_2_CAM, R_RECT_0):
    # (N, 3) -> (N, 3)
    N = pc_velo.shape[0]
    points = np.hstack([pc_velo, np.ones((N, 1))]).T
    points = np.matmul(T_VELO_2_CAM, points)
    points = np.matmul(R_RECT_0, points).T
    points = points[:, 0:3]
    pc_cam = points.reshape(-1, 3)
    return pc_cam


def load_calib(calib_path):
    CAM = 2
    lines = open(calib_path).readlines()
    lines = [ line.split()[1:] for line in lines ][:-1]
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


def angle2class(angle):
    angle %= 2 * np.pi
    if angle > 7*np.pi/4:
        angle -= 2 * np.pi
    angle_per_class = 2 * np.pi / NUM_ANGLE_BIN
    shifted_angle = angle + angle_per_class / 2
    angle_cls = int(shifted_angle / angle_per_class)
    angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
    return angle_cls, angle_res


def class2angle(angle_cls, angle_res):
    angle_per_class = 2 * np.pi / NUM_ANGLE_BIN
    angle_center = angle_cls * angle_per_class
    angle = angle_center + angle_res  # -pi/4 ~ 7*pi/4
    angle %= 2 * np.pi
    return angle

# def angle2class(angle):
#     angle %= np.pi
#     if angle > 3*np.pi/4:
#         angle -= np.pi
#     angle_per_class = np.pi / 2
#     shifted_angle = angle + angle_per_class / 2
#     angle_cls = int(shifted_angle / angle_per_class)
#     angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
#     return angle_cls, angle_res
#
#
# def class2angle(angle_cls, angle_res):
#     angle_per_class = np.pi / 2
#     angle_center = angle_cls * angle_per_class
#     angle = angle_center + angle_res  # -pi/4 ~ 3*pi/4
#     angle %= np.pi
#     return angle


def size2class(size):
    size_res = size - g_mean_size_arr
    return size_res


def class2size(size_res):
    size = g_mean_size_arr + size_res
    return size


def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    if len(pc.shape) == 1:
        pc = pc.reshape(-1, 3)
        pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
        pc = pc.reshape(3)
    else:
        pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc


def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def compute_box3d_iou(center_pred, angle_cls_pred, angle_res_pred, size_res_pred,
                      center_label, angle_cls_label, angle_res_label, size_res_label):

    batch_size = angle_cls_pred.shape[0]
    angle_cls = np.argmax(angle_cls_pred, 1) # B
    angle_res = np.array([angle_res_pred[i, angle_cls[i]] for i in range(batch_size)]) # B,
    size_res = np.vstack([size_res_pred[i, :] for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    for i in range(batch_size):
        heading_angle = class2angle(angle_cls[i], angle_res[i])
        box_size = class2size(size_res[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(angle_cls_label[i], angle_res_label[i])
        box_size_label = class2size(size_res_label[i])
        corners_3d_label = get_3d_box(box_size_label, heading_angle_label, center_label[i])
        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)

        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
        np.array(iou3d_list, dtype=np.float32)

# this just for visulize and testing
def cam_box3d_to_camera_box(box3d, P2):
    # h,w,l,x,y,z,ry -> x1,y1,x2,y2/8*(x, y)
    box_size, center, heading_angle = box3d[0:3], box3d[3:6], box3d[6]
    cam_box3d_corner = get_3d_box(box_size, heading_angle, center)

    points = np.hstack((cam_box3d_corner, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
    points = np.matmul(P2, points).T
    points[:, 0] /= points[:, 2]
    points[:, 1] /= points[:, 2]

    minx = np.clip(np.min(points[:, 0]), 0, 1241)
    maxx = np.clip(np.max(points[:, 0]), 0, 1241)
    miny = np.clip(np.min(points[:, 1]), 0, 374)
    maxy = np.clip(np.max(points[:, 1]), 0, 374)

    box2d = [minx, miny, maxx, maxy]

    return box2d



def write_results(calib_dir, result_dir, data_idx_list, center_list, angle_cls_list, angle_res_list,
                  size_res_list, score_list):
    results = {}
    template = '{} ' + ' '.join(['{:.2f}' for i in range(15)]) + '\n'

    for i in range(len(center_list)):
        idx = data_idx_list[i]
        calib_file = os.path.join(calib_dir, idx) + ".txt"
        P2, T_VELO_2_CAM, R_RECT_0 = load_calib(calib_file)

        cls = 'Car'
        score = score_list[i]
        x, y, z = center_list[i]
        l, w, h = class2size(size_res_list[i])

        ry = class2angle(angle_cls_list[i], angle_res_list[i])
        ry = wrapToPi(ry)
        # get box2d
        box = [l, w, h, x, y, z, ry]
        box2d = cam_box3d_to_camera_box(box, P2)

        y += h / 2.0
        box3d = [h, w, l, x, y, z, ry]

        if idx not in results:
            results[idx] = []
        result = template.format(cls, 0, 0, 0, *box2d, *box3d, float(score))
        # print(idx, result)
        results[idx].append(result)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%s.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line)
        fout.close()