import numpy as np

DIST_RANGE = 80

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
    
def filter(label_file, det_file):
    car_filtered_label_file = label_file.replace('.txt', '_car_filtered.txt')
    car_filtered_det_file = det_file.replace('.txt', '_car_filtered.txt')
    f_car_label_writer = open(car_filtered_label_file, 'w')
    f_car_det_writer = open(car_filtered_det_file, 'w')
    person_filtered_label_file = label_file.replace('.txt', '_person_filtered.txt')
    person_filtered_det_file = det_file.replace('.txt', '_person_filtered.txt')
    f_person_label_writer = open(person_filtered_label_file, 'w')
    f_person_det_writer = open(person_filtered_det_file, 'w')
    
    with open(det_file, 'r') as para:
        dets = para.read().strip('\n').split('\n')
    with open(label_file, 'r') as para:
        labels = para.read().strip('\n').split('\n')
        
    for det, label in zip(dets, labels):
        det_split = det.split(' ')
        img_id = det_split[0]
        cls = det_split[1]
        pc_velo_path = '/data/dataset/kitti_fvnet2/refinement/training/cropped/' + img_id + ".npy"
        pc_velo = np.load(pc_velo_path)
        front_box = np.array(det_split[2:8], dtype=np.float32)

        object_velo = crop_points(pc_velo, front_box)
        if len(object_velo) >= 5:
            if cls == 'Car':
              f_car_det_writer.write(det + '\n')
              f_car_label_writer.write(label + '\n')
            else:
              f_person_det_writer.write(det + '\n')
              f_person_label_writer.write(label + '\n')

    
    
if __name__ == '__main__':
    train_label_file = '/data/dataset/kitti_fvnet2/refinement/list_files/label_train.txt'
    train_det_file = '/data/dataset/kitti_fvnet2/refinement/list_files/det_train.txt'

    val_label_file = '/data/dataset/kitti_fvnet2/refinement/list_files/label_val.txt'
    val_det_file = '/data/dataset/kitti_fvnet2/refinement/list_files/det_val.txt'
    filter(train_label_file, train_det_file)
    filter(val_label_file, val_det_file)