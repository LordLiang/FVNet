from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from kitti_util import NUM_ANGLE_BIN, g_mean_size_arr
from model_util import parse_output_to_tensors, get_box3d_corners, get_box3d_corners_helper


def get_center_regression_net(object_point_cloud, is_training, bn_decay):
    ''' Regression network for center delta. a.k.a. T-Net.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in 3D mask coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        predicted_center: TF tensor in shape (B,3)
    '''
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3-stage1', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
        padding='VALID', scope='maxpool-stage1')
    net = tf.squeeze(net, axis=[1,2])
    net = tf_util.fully_connected(net, 256, scope='fc1-stage1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, scope='fc2-stage1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    predicted_center = tf_util.fully_connected(net, 3, activation_fn=None,
        scope='fc3-stage1')
    return predicted_center


def get_3d_box_estimation_net(object_point_cloud, is_training, bn_decay):
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='maxpool2')
    net = tf.squeeze(net, axis=[1, 2])
    net = tf_util.fully_connected(net, 512, scope='fc1', bn=True,
                                  is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, scope='fc2', bn=True,
                                  is_training=is_training, bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_ANGLE_BIN*2:  heading bin class scores and bin residuals
    # next 3: box cluster residuals
    output = tf_util.fully_connected(net,
                                     3 + NUM_ANGLE_BIN * 2 + 3, activation_fn=None, scope='fc3')
    return output


def get_model(point_cloud, is_training, bn_decay=None):
    end_points = {}

    # T-Net and coordinate translation
    center_delta = get_center_regression_net(point_cloud, is_training, bn_decay)
    end_points['center_delta'] = center_delta

    # Get object point cloud in object coordinate
    point_cloud_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])  # BxNx3
    point_cloud_features = tf.slice(point_cloud, [0, 0, 3], [-1, -1, -1])
    point_cloud_xyz_new = point_cloud_xyz - tf.expand_dims(center_delta, 1)
    point_cloud_new = tf.concat([point_cloud_xyz_new, point_cloud_features], axis=-1)

    # Amodel Box Estimation PointNet
    output = get_3d_box_estimation_net(point_cloud_new, is_training, bn_decay)

    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_res'] + end_points['center_delta']

    return end_points


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(losses)


def get_loss(center_label, angle_cls_label, angle_res_label, size_res_label, end_points):

    # Center regression losses
    x_dist = tf.norm(center_label[..., 0] - end_points['center'][..., 0], axis=-1)
    x_loss = huber_loss(x_dist, delta=1.0)
    y_dist = tf.norm(center_label[..., 1] - end_points['center'][..., 1], axis=-1)
    y_loss = huber_loss(y_dist, delta=1.0)
    z_dist = tf.norm(center_label[..., 2] - end_points['center'][..., 2], axis=-1)
    z_loss = huber_loss(z_dist, delta=1.0)
    center_loss = x_loss + y_loss + z_loss
    # center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    # center_loss = huber_loss(center_dist, delta=2.0)
    tf.summary.scalar('center_loss', center_loss)

    stage1_x_dist = tf.norm(center_label[..., 0] - end_points['center_delta'][..., 0], axis=-1)
    stage1_x_loss = huber_loss(stage1_x_dist, delta=1.0)
    stage1_y_dist = tf.norm(center_label[..., 1] - end_points['center_delta'][..., 1], axis=-1)
    stage1_y_loss = huber_loss(stage1_y_dist, delta=1.0)
    stage1_z_dist = tf.norm(center_label[..., 2] - end_points['center_delta'][..., 2], axis=-1)
    stage1_z_loss = huber_loss(stage1_z_dist, delta=1.0)

    # stage1_center_dist = tf.norm(center_label - end_points['center_delta'], axis=-1)
    # stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    stage1_center_loss = stage1_x_loss + stage1_y_loss + stage1_z_loss
    tf.summary.scalar('stage1 center loss', stage1_center_loss)

    # Heading angle loss
    angle_cls_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['angle_scores'], labels=angle_cls_label))
    tf.summary.scalar('angle_class_loss', angle_cls_loss)

    hcls_onehot = tf.one_hot(angle_cls_label,
                             depth=NUM_ANGLE_BIN,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_ANGLE_BIN
    angle_per_class = 2 * np.pi / NUM_ANGLE_BIN
    angle_res_normalized_label = angle_res_label / (angle_per_class / 2)
    angle_res_normalized_loss = huber_loss(tf.reduce_sum( \
        end_points['angle_res_normalized'] * tf.to_float(hcls_onehot), axis=1) - \
                                                  angle_res_normalized_label, delta=1.0)
    tf.summary.scalar('angle_res_loss', angle_res_normalized_loss)

    # Size loss
    mean_sizes = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # (1,NS,3)

    size_res_label_normalized = size_res_label / mean_sizes
    # size_normalized_dist = tf.norm( \
    #     size_res_label_normalized - end_points['size_res_normalized'],
    #     axis=-1)
    l_dist = tf.norm(size_res_label_normalized[..., 0] - end_points['size_res_normalized'][..., 0], axis=-1)
    l_loss = huber_loss(l_dist, delta=1.0)
    w_dist = tf.norm(size_res_label_normalized[..., 1] - end_points['size_res_normalized'][..., 1], axis=-1)
    w_loss = huber_loss(w_dist, delta=1.0)
    h_dist = tf.norm(size_res_label_normalized[..., 2] - end_points['size_res_normalized'][..., 2], axis=-1)
    h_loss = huber_loss(h_dist, delta=1.0)
    # size_res_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    size_res_normalized_loss = l_loss + w_loss + h_loss
    tf.summary.scalar('size_res_loss', size_res_normalized_loss)

    # Corner loss
    corners_3d = get_box3d_corners(end_points['center'], end_points['angle_res'], end_points['size_res'])  # (B,NH,8,3)

    corners_3d_pred = tf.reduce_sum( \
        tf.to_float(tf.expand_dims(tf.expand_dims(hcls_onehot, -1), -1)) * corners_3d,
        axis=[1])  # (B,8,3)

    angle_bin_centers = tf.constant( \
        np.arange(0, 2*np.pi, 2 * np.pi/NUM_ANGLE_BIN), dtype=tf.float32)  # (NH,)
    heading_label = tf.expand_dims(angle_res_label, 1) + \
                    tf.expand_dims(angle_bin_centers, 0)  # (B,NH)
    heading_label = tf.reduce_sum(tf.to_float(hcls_onehot) * heading_label, 1)

    size_label = mean_sizes + size_res_label

    corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)  # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper( \
        center_label, heading_label + np.pi, size_label)  # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1),
                              tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    # corners_dist = tf.norm(corners_3d_pred - corners_3d_gt, axis=-1)
    corners_loss = huber_loss(corners_dist, delta=1.0)
    tf.summary.scalar('corners_loss', corners_loss)

    total_loss = center_loss + \
               stage1_center_loss + \
               angle_cls_loss + \
               20.0 * angle_res_normalized_loss + \
               20.0 * size_res_normalized_loss + \
               5.0 * corners_loss


    tf.add_to_collection('losses', total_loss)

    return [total_loss, center_loss, stage1_center_loss, angle_cls_loss, \
           angle_res_normalized_loss, size_res_normalized_loss, corners_loss]
