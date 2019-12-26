from __future__ import print_function

import os
import sys
import time
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))

from kitti_dataset import KittiDataset, get_batch
from kitti_util import compute_box3d_iou, write_results


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--model', default='front_pointnets_v1', help='Model name')
parser.add_argument('--output_dir', default='../prediction/val', help='Log dir')
parser.add_argument('--num_point', type=int, default=512, help='Point Number')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument('--model_path', default='../model_1000.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch_size
NUM_CHANNEL = 4

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_PATH = FLAGS.model_path

RESULT_DIR = FLAGS.output_dir
if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
LOG_FOUT = open(os.path.join(RESULT_DIR, 'log_test_val.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

prefix = "/data/dataset/est-kitti/"
DATA_DIR = prefix + "training/"
LIST_FILE = prefix + "list_files/det_val_car_filtered.txt"
LABEL_FILE = prefix + "list_files/label_val_2_car_filtered.txt"
VAL_DATASET = KittiDataset(NUM_POINT, DATA_DIR, LIST_FILE, LABEL_FILE)
TEST_DATASET = KittiDataset(NUM_POINT, DATA_DIR, LIST_FILE, is_test=True)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def val():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):

            pointcloud_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, NUM_CHANNEL))
            center_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 3))
            angle_cls_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
            angle_res_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE,))
            size_res_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 3))

            is_training_pl = tf.placeholder(tf.bool, shape=())

            end_points = MODEL.get_model(pointcloud_pl, is_training_pl)
            loss_list = MODEL.get_loss(center_pl, angle_cls_pl, angle_res_pl, size_res_pl, end_points)
            loss, center_loss, stage1_center_loss, h_cls_loss, \
            h_res_loss, s_res_loss, corners_loss = loss_list
            tf.summary.scalar('loss', loss)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

            # Write summaries of bounding box IoU accuracies
            iou2ds, iou3ds = tf.py_func(compute_box3d_iou, [
                end_points['center'],
                end_points['angle_scores'], end_points['angle_res'],
                end_points['size_res'],
                center_pl, angle_cls_pl, angle_res_pl, size_res_pl],
                [tf.float32, tf.float32])

            end_points['iou2ds'] = iou2ds
            end_points['iou3ds'] = iou3ds
            tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
            tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)

        ops = {'pointcloud_pl': pointcloud_pl,
               'center_pl': center_pl,
               'angle_cls_pl': angle_cls_pl,
               'angle_res_pl': angle_res_pl,
               'size_res_pl': size_res_pl,
               'is_training_pl': is_training_pl,
               'total_loss': total_loss,
               'center_loss': center_loss,
               'stage1_center_loss': stage1_center_loss,
               'h_cls_loss': h_cls_loss,
               'h_res_loss': h_res_loss,
               's_res_loss': s_res_loss,
               'corners_loss': corners_loss,
               'end_points': end_points}

        eval_one_epoch(sess, ops)
        inference(sess, ops)


def eval_one_epoch(sess, ops):
    val_idxs = np.arange(0, len(VAL_DATASET))
    num_batches = int(np.ceil(len(VAL_DATASET) / BATCH_SIZE))

    # To collect statistics

    total_loss_sum = 0
    center_loss_sum = 0
    stage1_center_loss_sum = 0
    h_cls_loss_sum = 0
    h_res_loss_sum = 0
    s_res_loss_sum = 0
    corners_loss_sum = 0

    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt_50 = 0
    iou3d_correct_cnt_70 = 0

    # Simple evaluation with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_center, \
        batch_angle_cls, batch_angle_res, batch_size_res = \
            get_batch(VAL_DATASET, val_idxs, start_idx, end_idx, NUM_CHANNEL)

        feed_dict = {ops['pointcloud_pl']: batch_data,
                     ops['center_pl']: batch_center,
                     ops['angle_cls_pl']: batch_angle_cls,
                     ops['angle_res_pl']: batch_angle_res,
                     ops['size_res_pl']: batch_size_res,
                     ops['is_training_pl']: False}

        ep = ops['end_points']
        total_loss_val, center_loss_val, stage1_center_loss_val, h_cls_loss_val, \
        h_res_loss_val, s_res_loss_val, corners_loss_val, \
        iou2ds, iou3ds = \
            sess.run([ops['total_loss'],
                      ops['center_loss'], ops['stage1_center_loss'], ops['h_cls_loss'],
                      ops['h_res_loss'], ops['s_res_loss'], ops['corners_loss'],
                      ep['iou2ds'], ep['iou3ds']],
                     feed_dict=feed_dict)

        total_loss_sum += total_loss_val
        center_loss_sum += center_loss_val
        stage1_center_loss_sum += stage1_center_loss_val
        h_cls_loss_sum += h_cls_loss_val
        h_res_loss_sum += h_res_loss_val
        s_res_loss_sum += s_res_loss_val
        corners_loss_sum += corners_loss_val

        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt_50 += np.sum(iou3ds >= 0.5)
        iou3d_correct_cnt_70 += np.sum(iou3ds >= 0.7)

    log_string('eval mean total loss: %f' % (total_loss_sum / float(num_batches)))
    log_string('eval mean center loss: %f' % (center_loss_sum / float(num_batches)))
    log_string('eval mean stage1 center loss: %f' % (stage1_center_loss_sum / float(num_batches)))
    log_string('eval mean angle class loss: %f' % (h_cls_loss_sum / float(num_batches)))
    log_string('eval mean angle res loss: %f' % (h_res_loss_sum / float(num_batches)))
    log_string('eval mean size res loss: %f' % (s_res_loss_sum / float(num_batches)))
    log_string('eval mean corners loss: %f' % (corners_loss_sum / float(num_batches)))

    log_string('eval box IoU (ground/3D): %f / %f' % \
               (iou2ds_sum / float(num_batches * BATCH_SIZE), iou3ds_sum / \
                float(num_batches * BATCH_SIZE)))
    log_string('eval box estimation accuracy (IoU=0.5): %f' % \
               (float(iou3d_correct_cnt_50) / float(num_batches * BATCH_SIZE)))
    log_string('eval box estimation accuracy (IoU=0.7): %f' % \
               (float(iou3d_correct_cnt_70) / float(num_batches * BATCH_SIZE)))


def inference(sess, ops):
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = int(np.ceil(len(TEST_DATASET) / BATCH_SIZE))
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data_idx, batch_data, batch_xyz_mean, batch_prob = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx, NUM_CHANNEL)

        feed_dict = {ops['pointcloud_pl']: batch_data,
                     ops['is_training_pl']: False}

        ep = ops['end_points']
        batch_center, batch_angle_scores, batch_angle_residuals, batch_size_residuals = \
            sess.run([ep['center'],
                      ep['angle_scores'], ep['angle_res'],
                      ep['size_res']],
                     feed_dict=feed_dict)

        batch_center += batch_xyz_mean
        batch_angle_cls = np.argmax(batch_angle_scores, 1)
        batch_angle_res = np.array([batch_angle_residuals[i, batch_angle_cls[i]] for i in range(BATCH_SIZE)])
        batch_size_res = np.vstack([batch_size_residuals[i] for i in range(BATCH_SIZE)])

        angle_prob = np.max(softmax(batch_angle_scores), 1)
        batch_scores = (angle_prob + batch_prob) / 2

        TEST_CALIB_DIR = "/data/dataset/KITTI/object/training/calib"
        write_results(TEST_CALIB_DIR, RESULT_DIR,
                      batch_data_idx, batch_center,
                      batch_angle_cls, batch_angle_res,
                      batch_size_res, batch_scores)

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    val()
    LOG_FOUT.close()
