import os
import numpy as np
import math
from PIL import Image
import json
import time
from xml.dom.minidom import Document

H_MIN = -3
H_MAX = 1

D_MIN = 0
D_MAX = 80

DATA_DIR = '/data/dataset/KITTI/object/'
PRO_DIR = '/data/dataset/kitti_fvnet/projection/'
REF_DIR = '/data/dataset/kitti_fvnet/refinement/'

cats = ['Car', 'Person'] # you can use ['Car', 'Pedestrian', 'Cyclist'] for 3 catelories
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
cat_info = []
for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i + 1})

def preprocess(split_list, has_label=False, tag='train'):
    start = time.time()
    
    ret = {'images': [], 'annotations': [], "categories": cat_info}

    image_to_id = {}
    if tag in ['train', 'val']:
        folder = 'training'
    else:
        folder = 'testing'
    data_folder = DATA_DIR + folder

    if has_label:
        det_list_file = REF_DIR + 'list_files/det_' + tag + '.txt'
        if os.path.exists(det_list_file):
            os.remove(det_list_file)
        est_list_file = REF_DIR + 'list_files/label_' + tag + '.txt'
        if os.path.exists(est_list_file):
            os.remove(est_list_file)

    for line in open(split_list):
        if line[-1] == '\n':
            line = line[:-1]
        image_id = int(line)
        print('preprocessing', tag, image_id)
        image_info = {'file_name': '{}.png'.format(line), 'id': image_id}
        ret['images'].append(image_info)
        
        # crop pointcloud
        pc_path = os.path.join(data_folder, 'velodyne', line + '.bin')
        calib_path = os.path.join(data_folder, 'calib', line + '.txt')

        pts = load_velodyne_points(pc_path)
        pts = filter_height(pts)
        P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_path)
        idx = crop_camera(pts, P, Tr_velo_to_cam, R_cam_to_rect)
        pts = pts[idx]
        # crop point clouds for refinement network, of course you can use raw .bin for training
        points_save_path = REF_DIR + folder + '/cropped/' + line + '.npy'
        np.save(points_save_path, pts)
        # just for visualization, you can open .xyz files by MeshLab software directly
        points_save_path = REF_DIR + folder + '/vis/' + line + '.xyz'
        np.savetxt(points_save_path, pts[:,0:3])

        # label
        if has_label:
            label_path = os.path.join(data_folder, 'label_2', line + '.txt')
            bounding_boxes, labels = load_label(label_path, Tr_velo_to_cam, R_cam_to_rect)
            front_map, boxes_2d = get_map_and_boxes2d(pts, True, bounding_boxes)
            xml_labels = []

            if len(boxes_2d) > 0:
                for id in range(len(boxes_2d)):
                    a1, a2, z1, z2, d1, d2 = boxes_2d[id]
                    x1, x2, y1, y2 = 1-a2, 1-a1, 1-z2, 1-z1
                    x1, y1 = int(x1 * 512), int(y1 * 128)
                    x2, y2 = math.ceil(x2 * 512), math.ceil(y2 * 128)
                    D1, D2 = float('%.5f' % (d1 * 80)), float('%.5f' % (d2 * 80))
                    class_label, truncated, occluded = labels[id].split()[:3]
                    truncated, occluded = int(float(truncated)), int(occluded)
                    # if you use 3 catelories, you don't need so this step
                    if class_label in ['Pedestrian', 'Cyclist']:
                        class_label = 'Person'
                    if class_label in cats and D1 < 80:
                        cat_id = cat_ids[class_label]
                        iscrowd = int(occluded != 0)
                        ann = {'image_id': image_id,
                               'id': int(len(ret['annotations']) + 1),
                               'category_id': cat_id,
                               'bbox': [x1, y1, x2-x1, y2-y1],
                               'depth':[D1, D2],
                               'truncated': truncated,
                               'occluded': occluded,
                               'iscrowd': 0,
                               'ignore': 0,
                               'area': (x2-x1)*(y2-y1),
                               'segmentation': []}
                        ret['annotations'].append(ann)
                        xml_labels.append([class_label, truncated, occluded, x1, x2, y1, y2, D1, D2])

                        with open(det_list_file, 'a') as file:
                            file.write('%s %s %.5f %.5f %.5f %.5f %.5f %.5f\n' % (line, class_label, a1, a2, z1, z2, d1, d2))
    
                        with open(est_list_file, 'a') as file:
                            file.write(line + ' ' + labels[id] + '\n')
                            
            xml_save_path = PRO_DIR + folder + '/VOCdevkit/Annotations/' + line + '.xml'
            writeInfoToXml(xml_labels, xml_save_path, line)
        else:
            front_map, _ = get_map_and_boxes2d(pts, True)
        
        if front_map is not None:
            front_map_png = Image.fromarray(front_map)
            front_png_save_path = PRO_DIR + folder + '/images/' + line + '.png'
            front_map_png.save(front_png_save_path, 'png')
            voc_png_save_path = PRO_DIR + folder + '/VOCdevkit/PNGImages/' + line + '.png'
            front_map_png.save(voc_png_save_path, 'png')
            
    end = time.time()
        
    
    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    print("used time:%.2fs"%(end-start))    
    out_path = '{}/annotations/kitti_{}.json'.format(PRO_DIR+folder, tag)
    json.dump(ret, open(out_path, 'w'))


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

# need further optimization for faster processing speed, such as C++ code
def get_map_and_boxes2d(points, return_map=True, bounding_boxes=None):
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
    d_range = D_MAX - D_MIN
    h_range = H_MAX - H_MIN
    
    if return_map:

        height_ratio = (z - H_MIN) / h_range
        dist_ratio = (plane_dist - D_MIN) / d_range
        # here I use a fusion of 3 front maps to generate the final front map, and you can just use one of them
        # 128x512
        front_map_0 = np.zeros((128, 512, 3))
        index_h_0 = np.floor((zenith - Z_MIN) / z_range * 128).astype(np.int)
        index_w_0 = np.floor((azimuth - A_MIN) / a_range * 512).astype(np.int)
        # 64x256
        front_map_1 = np.zeros((64, 256, 3))
        index_h_1 = np.floor((zenith - Z_MIN) / z_range * 64).astype(np.int)
        index_w_1 = np.floor((azimuth - A_MIN) / a_range * 256).astype(np.int)
        # 32x128
        front_map_2 = np.zeros((32, 128, 3))
        index_h_2 = np.floor((zenith - Z_MIN) / z_range * 32).astype(np.int)
        index_w_2 = np.floor((azimuth - A_MIN) / a_range * 128).astype(np.int)
        
        
        for i in range(len(points)):
            if (index_h_0[i] < 128) and (index_w_0[i] < 512):
                dr_0 = front_map_0[index_h_0[i], index_w_0[i], 1]
                if (dr_0 == 0) or (dist_ratio[i] < dr_0):
                    front_map_0[index_h_0[i], index_w_0[i], 2] = height_ratio[i]  # height ratio 0-1
                    front_map_0[index_h_0[i], index_w_0[i], 1] = dist_ratio[i]  # dist ratio 0-1
                    front_map_0[index_h_0[i], index_w_0[i], 0] = intensity[i]  # intensity 0-1
                
            if (index_h_1[i] < 64) and (index_w_1[i] < 256):
                dr_1 = front_map_1[index_h_1[i], index_w_1[i], 1]
                if (dr_1 == 0) or (dist_ratio[i] < dr_1):
                    front_map_1[index_h_1[i], index_w_1[i], 2] = height_ratio[i]  # height ratio 0-1
                    front_map_1[index_h_1[i], index_w_1[i], 1] = dist_ratio[i]  # dist ratio 0-1
                    front_map_1[index_h_1[i], index_w_1[i], 0] = intensity[i]  # intensity 0-1
                        
            if (index_h_2[i] < 32) and (index_w_2[i] < 128):
                dr_2 = front_map_2[index_h_2[i], index_w_2[i], 1]
                if (dr_2 == 0) or (dist_ratio[i] < dr_2):
                    front_map_2[index_h_2[i], index_w_2[i], 2] = height_ratio[i]  # height ratio 0-1
                    front_map_2[index_h_2[i], index_w_2[i], 1] = dist_ratio[i]  # dist ratio 0-1
                    front_map_2[index_h_2[i], index_w_2[i], 0] = intensity[i]  # intensity 0-1          
    
        front_map_0 = FZ(front_map_0)
        front_map_0 = (front_map_0 * 255).astype(np.uint8)
            
        front_map_1 = FZ(front_map_1)
        front_map_1 = (front_map_1 * 255).astype(np.uint8)
        front_map_png_1 = Image.fromarray(front_map_1)
        front_map_png_1 = front_map_png_1.resize((512, 128), Image.NEAREST)
        front_map_1 = np.array(front_map_png_1)
            
        front_map_2 = FZ(front_map_2)
        front_map_2 = (front_map_2 * 255).astype(np.uint8)
        front_map_png_2 = Image.fromarray(front_map_2)
        front_map_png_2 = front_map_png_2.resize((512, 128), Image.NEAREST)
        front_map_2 = np.array(front_map_png_2)
    
        # fusion
        front_map = front_map_0
        mask = front_map == np.array([0,0,0])
        front_map[mask] = front_map_1[mask]
        mask = front_map == np.array([0,0,0])
        front_map[mask] = front_map_2[mask]
    else:
        front_map = None

    if bounding_boxes is not None:
        corner_boxes = get_corner_boxes(bounding_boxes)
        boxes_2d = []
        for box3d in corner_boxes:
            corner_x = box3d[:, 0]
            corner_y = box3d[:, 1]
            corner_z = box3d[:, 2]
            corner_space_dist = np.sqrt(corner_x * corner_x + corner_y * corner_y + corner_z * corner_z)
            corner_plane_dist = np.sqrt(corner_x * corner_x + corner_y * corner_y)
            corner_azimuth = np.arcsin(corner_y / corner_plane_dist)
            corner_zenith = np.arcsin(corner_z / corner_space_dist)
            x1 = (corner_azimuth.min() - A_MIN) / a_range
            x2 = (corner_azimuth.max() - A_MIN) / a_range
            y1 = (corner_zenith.min() - Z_MIN) / z_range
            y2 = (corner_zenith.max() - Z_MIN) / z_range
            d1 = (corner_plane_dist.min() - D_MIN) / d_range
            d2 = (corner_plane_dist.max() - D_MIN) / d_range

            boxes_2d.append([x1, x2, y1, y2, d1, d2])

        boxes_2d = np.array(boxes_2d, dtype=np.float32)
        boxes_2d = np.clip(boxes_2d, 0.0, 1.0)
    else:
        boxes_2d = None

    return front_map, boxes_2d


def load_velodyne_points(pc_path):
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    return points


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


def load_label(label_path, T_VELO_2_CAM, R_RECT_0):
    bounding_boxes = []
    care_labels = []
    with open(label_path, 'r') as para:
        labels = para.read().strip('\n').split('\n')
        for label in labels:
            label_split = label.split(' ')
            if label_split[0] != 'DontCare':
                bounding_boxes.append(label_split[8:15])
                care_labels.append(label)     
    if len(bounding_boxes)>0:
        bounding_boxes = np.array(bounding_boxes, dtype=np.float32)
        center_coords = bounding_boxes[:, 3:6]
        bounding_boxes[:, 3:6] = camera_to_lidar(center_coords, T_VELO_2_CAM, R_RECT_0)
    return bounding_boxes, care_labels



def camera_to_lidar(points, T_VELO_2_CAM, R_RECT_0):
    N = points.shape[0]
    points_ext = np.hstack([points, np.ones((N, 1))]).T
    points_ext = np.matmul(np.linalg.inv(R_RECT_0), points_ext)
    points_ext = np.matmul(np.linalg.inv(T_VELO_2_CAM), points_ext).T
    points_ext = points_ext[:, 0:3]
    return points_ext


def get_corner_boxes(center_boxes):
    corner_boxes = []
    for box in center_boxes:
        h, w, l, x, y, z, ry = box

        rz = - np.pi / 2 - ry
        rotate_matrix = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        trans_matrix = np.array([x, y, z])

        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        cornerPosInVelo = np.dot(rotate_matrix, trackletBox) + \
                          np.tile(trans_matrix, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        corner_boxes.append(box3d)

    return corner_boxes


def filter_height(points):
    idx = np.logical_and(np.logical_and(points[:, 2] >= H_MIN, points[:, 2] <= H_MAX),
                         np.logical_and(points[:, 0] >= 0, points[:, 0] <= 80))
    return points[idx]

def writeInfoToXml(labels, save_path, line):

    doc = Document()
    annolist = doc.createElement('annotation')
    doc.appendChild(annolist)

    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(line + '.png')
    filename.appendChild(filename_text)
    annolist.appendChild(filename)

    for i in range(len(labels)):
        class_label, truncated_val, occluded, x1, x2, y1, y2, D1, D2 = labels[i]

        obj = doc.createElement('object')
        annolist.appendChild(obj)

        name = doc.createElement('name')
        name_text = doc.createTextNode(class_label)
        name.appendChild(name_text)
        obj.appendChild(name)

        truncated = doc.createElement('truncated')
        truncated_text = doc.createTextNode(str(truncated_val))
        truncated.appendChild(truncated_text)
        obj.appendChild(truncated)

        difficult = doc.createElement('difficult')
        difficult_text = doc.createTextNode(str(occluded))
        difficult.appendChild(difficult_text)
        obj.appendChild(difficult)

        front_depth = doc.createElement('front_depth')
        front_text = doc.createTextNode(str(D1))
        front_depth.appendChild(front_text)
        obj.appendChild(front_depth)

        back_depth = doc.createElement('back_depth')
        back_text = doc.createTextNode(str(D2))
        back_depth.appendChild(back_text)
        obj.appendChild(back_depth)

        bndbox = doc.createElement('bndbox')
        obj.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        xmin_text = doc.createTextNode(str(x1))
        xmin.appendChild(xmin_text)
        bndbox.appendChild(xmin)

        ymin = doc.createElement('ymin')
        ymin_text = doc.createTextNode(str(y1))
        ymin.appendChild(ymin_text)
        bndbox.appendChild(ymin)

        xmax = doc.createElement('xmax')
        xmax_text = doc.createTextNode(str(x2))
        xmax.appendChild(xmax_text)
        bndbox.appendChild(xmax)

        ymax = doc.createElement('ymax')
        ymax_text = doc.createTextNode(str(y2))
        ymax.appendChild(ymax_text)
        bndbox.appendChild(ymax)

    f = open(save_path, 'w')
    doc.writexml(f, indent='\t', newl='\n', addindent='\t')
    f.close()

    return
    
if __name__ == '__main__':
    preprocess('image_sets/train.txt', has_label=True, tag='train')
    preprocess('image_sets/val.txt', has_label=True, tag='val')
    preprocess('image_sets/test.txt', has_label=False, tag='test')

