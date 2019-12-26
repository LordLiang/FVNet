import numpy as np

from kitti_util import random_perturb_box, crop_points, load_calib, lidar_to_camera_point
from kitti_util import angle2class, size2class, wrapToPi, rotate_pc_along_y

class KittiDataset(object):
    def __init__(self, num_point, data_dir, det_file, label_file=None,
                 perturb_box=False, aug=False, is_test=False):
        self.img_dir = data_dir + "/image_2/"
        self.calib_dir = data_dir + "/calib/"
        self.lidar_dir = data_dir + "/cropped/"
        self.num_point = num_point
        self.data_dir = data_dir
        self.aug = aug
        self.perturb_box = perturb_box
        self.is_test = is_test
        with open(det_file, 'r') as para:
            self.dets = para.read().strip('\n').split('\n')
        if not self.is_test:
            with open(label_file, 'r') as para:
                    self.labels = para.read().strip('\n').split('\n')

    def __getitem__(self, index):
        det = self.dets[index].split(' ')
        img_id = det[0]
        pc_velo_path = self.lidar_dir + img_id + ".npy"
        pc_velo = np.load(pc_velo_path)
        front_box = np.array(det[2:8], dtype=np.float32)
        if self.perturb_box:
            new_front_box = random_perturb_box(front_box)
            object_velo = crop_points(pc_velo, new_front_box)
            if len(object_velo) < 5:
                object_velo = crop_points(pc_velo, front_box)
        else:
            object_velo = crop_points(pc_velo, front_box)

        calib_path = self.calib_dir + img_id + ".txt"
        P2, Tr_velo_to_cam, R0_rect = load_calib(calib_path)
        object_rect = np.zeros_like(object_velo)
        object_rect[:, 0:3] = lidar_to_camera_point(object_velo[:, 0:3], Tr_velo_to_cam, R0_rect)
        object_rect[:, 3] = object_velo[:, 3]
        xyz_mean = np.mean(object_rect[:, 0:3], axis=0)
        object_rect[:, 0:3] -= xyz_mean

        mask = np.random.choice(object_rect.shape[0],
                                self.num_point,
                                replace=(object_rect.shape[0] < self.num_point))
        object_rect = object_rect[mask, :]

        if self.is_test:
            # prob = float(det[9])
            prob = 1.0
            sample = {"img_id": img_id, "point_set": object_rect,
                      "xyz_mean": xyz_mean, "prob": prob}
        else:
            label = self.labels[index].split(' ')
            h, w, l, x, y, z, ry = np.array(label[9:], dtype=np.float32)
            y -= h/2
            size = np.array([l, w, h])
            center = np.array([x, y, z]) - xyz_mean

            if self.aug:
                flip = np.random.randint(low=0, high=2)
                object_rect[:, 0] = flip * (-object_rect[:, 0]) + (1 - flip) * object_rect[:, 0]
                center[0] = flip * (-center[0]) + (1 - flip) * center[0]
                if flip == 1:
                    ry = wrapToPi(np.pi - ry)

                # random_angle = np.random.uniform(low=-np.pi / 10, high=np.pi / 10)
                # object_rect_temp = object_rect.copy()
                # object_rect_temp[:, 0:3] = object_rect[:, 0:3] - center
                # object_rect_temp = rotate_pc_along_y(object_rect_temp, random_angle)
                # object_rect[:, 0:3] = object_rect_temp[:, 0:3] + center
                # ry = wrapToPi(ry + random_angle)

            angle_cls, angle_res = angle2class(ry)
            size_res = size2class(size)
            sample = {"point_set": object_rect,
                      "xyz_mean": xyz_mean, "center": center,
                      "angle_cls": angle_cls, "angle_res": angle_res,
                      "size_res": size_res}
        return sample


    def __len__(self):
        return len(self.dets)


def get_batch(dataset, idxs, start_idx, end_idx, num_channel):
    bs = end_idx - start_idx
    num_point = dataset.num_point
    batch_idx = []
    batch_data = np.zeros((bs, num_point, num_channel))
    num = len(dataset)

    if dataset.is_test:
        batch_xyz_mean = np.zeros((bs, 3))
        batch_prob = np.zeros((bs,))
        for i in range(bs):
            sample = dataset[idxs[(i + start_idx) % num]]
            batch_idx.append(sample["img_id"])
            batch_data[i] = sample["point_set"][:, 0:num_channel]
            batch_xyz_mean[i] = sample["xyz_mean"]
            batch_prob[i] = sample["prob"]
        return batch_idx, batch_data, batch_xyz_mean, batch_prob

    else:
        batch_center = np.zeros((bs, 3))
        batch_angle_cls = np.zeros((bs,), dtype=np.int32)
        batch_angle_res = np.zeros((bs,))
        batch_size_res = np.zeros((bs, 3))
        for i in range(bs):
            sample = dataset[idxs[(i + start_idx) % num]]
            batch_data[i] = sample["point_set"][:, 0:num_channel]
            batch_center[i] = sample["center"]
            batch_angle_cls[i] = sample["angle_cls"]
            batch_angle_res[i] = sample["angle_res"]
            batch_size_res[i] = sample["size_res"]

        return batch_data, batch_center,\
               batch_angle_cls, batch_angle_res, batch_size_res
