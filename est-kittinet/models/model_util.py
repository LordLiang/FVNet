import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
from kitti_util import NUM_ANGLE_BIN, g_mean_size_arr

NUM_OBJECT_POINT = 512


def parse_output_to_tensors(output, end_points):
    batch_size = output.get_shape()[0].value
    center_res = tf.slice(output, [0, 0], [-1, 3])

    end_points['center_res'] = center_res

    angle_scores = tf.slice(output, [0, 3], [-1, NUM_ANGLE_BIN])
    angle_res_norm = tf.slice(output, [0, 3 + NUM_ANGLE_BIN],
                                            [-1, NUM_ANGLE_BIN])
    end_points['angle_scores'] = angle_scores  # BxNUM_ANGLE_BIN
    end_points['angle_res_normalized'] = angle_res_norm  # BxNUM_ANGLE_BIN (-1 to 1)
    angle_per_class = 2 * np.pi / NUM_ANGLE_BIN
    end_points['angle_res'] = angle_res_norm * (angle_per_class / 2)  # BxNUM_ANGLE_BIN

    size_res_norm = tf.slice(output, [0, 3 + NUM_ANGLE_BIN * 2], [-1, 3])
    size_res_norm = tf.reshape(size_res_norm, [batch_size, 3])  # Bx3
    end_points['size_res_normalized'] = size_res_norm
    end_points['size_res'] = size_res_norm * tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)

    return end_points


def get_box3d_corners(center, angle_res, size_res):
    batch_size = center.get_shape()[0].value
    angle_bin_centers = tf.constant(np.arange(0, 2*np.pi, 2 * np.pi / NUM_ANGLE_BIN), dtype=tf.float32)  # (NH,)
    angles = angle_res + tf.expand_dims(angle_bin_centers, 0)  # (B,NH)

    mean_sizes = tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # (1,3)
    sizes = mean_sizes + size_res  # (B,3)
    sizes = tf.tile(tf.expand_dims(sizes, 1), [1, NUM_ANGLE_BIN, 1])  # (B,NH,3)

    centers = tf.tile(tf.expand_dims(center, 1), [1, NUM_ANGLE_BIN, 1])  # (B,NH,3)

    N = batch_size * NUM_ANGLE_BIN
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N, 3]), tf.reshape(angles, [N]),
                                          tf.reshape(sizes, [N, 3]))

    return tf.reshape(corners_3d, [batch_size, NUM_ANGLE_BIN, 8, 3])


def get_box3d_corners_helper(centers, angles, sizes):
    N = centers.get_shape()[0].value
    l = tf.slice(sizes, [0, 0], [-1, 1])  # (N,1)
    w = tf.slice(sizes, [0, 1], [-1, 1])  # (N,1)
    h = tf.slice(sizes, [0, 2], [-1, 1])  # (N,1)
    # print l,w,h
    x_corners = tf.concat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=1)  # (N,8)
    y_corners = tf.concat([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], axis=1)  # (N,8)
    z_corners = tf.concat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=1)  # (N,8)
    corners = tf.concat([tf.expand_dims(x_corners, 1), tf.expand_dims(y_corners, 1), tf.expand_dims(z_corners, 1)],
                        axis=1)  # (N,3,8)
    # print x_corners, y_corners, z_corners
    c = tf.cos(angles)
    s = tf.sin(angles)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)
    row1 = tf.stack([c, zeros, s], axis=1)  # (N,3)
    row2 = tf.stack([zeros, ones, zeros], axis=1)
    row3 = tf.stack([-s, zeros, c], axis=1)
    R = tf.concat([tf.expand_dims(row1, 1), tf.expand_dims(row2, 1), tf.expand_dims(row3, 1)], axis=1)  # (N,3,3)
    # print row1, row2, row3, R, N
    corners_3d = tf.matmul(R, corners)  # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers, 2), [1, 1, 8])  # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0, 2, 1])  # (N,8,3)
    return corners_3d




