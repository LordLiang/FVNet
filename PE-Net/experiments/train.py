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
from kitti_util import compute_box3d_iou


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--model', default='front_pointnets_v1', help='Model name')
parser.add_argument('--output_dir', default='../outputs', help='Log dir')
parser.add_argument('--num_point', type=int, default=512, help='Point Number')
parser.add_argument('--max_epoch', type=int, default=1001, help='Epoch to run')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate')
parser.add_argument('--optimizer', default='adam', help='adam or momentum')
parser.add_argument('--decay_step', type=int, default=800000, help='Decay step for lr decay')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
MAX_EPOCH = FLAGS.max_epoch
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
BASE_LEARNING_RATE = FLAGS.learning_rate
BATCH_SIZE = FLAGS.batch_size
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 4

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
LOG_DIR = os.path.join(FLAGS.output_dir, time_stamp)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp -r ../experiments/ ../kitti/ ../models/ ../kitti_eval %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

prefix = "/data/dataset/est-kitti/"
DATA_DIR = prefix + "training/"
TRAIN_LIST_FILE = prefix + "list_files/det_train_car_filtered.txt"
TRAIN_LABEL_FILE = prefix + "list_files/label_train_2_car_filtered.txt"
TRAIN_DATASET = KittiDataset(NUM_POINT, DATA_DIR, TRAIN_LIST_FILE, TRAIN_LABEL_FILE, perturb_box=True, aug=True)

VAL_LIST_FILE = prefix + "list_files/det_val_car_filtered.txt"
VAL_LABEL_FILE = prefix + "list_files/label_val_2_car_filtered.txt"
VAL_DATASET = KittiDataset(NUM_POINT, DATA_DIR, VAL_LIST_FILE, VAL_LABEL_FILE)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):

            pointcloud_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, NUM_CHANNEL))
            center_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 3))
            angle_cls_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))
            angle_res_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE,))
            size_res_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 3))

            is_training_pl = tf.placeholder(tf.bool, shape=())
            batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            end_points = MODEL.get_model(pointcloud_pl, is_training_pl, bn_decay=bn_decay)
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

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                       momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

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
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)

            # Save the variables to disk.
            if epoch > 0 and epoch % 5 == 0:
                eval_one_epoch(sess, ops, test_writer)
            if epoch > 0 and epoch % 20 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_"+ str(epoch) + ".ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    is_training = True
    log_string(str(datetime.now()))

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = int(np.ceil(len(TRAIN_DATASET) / BATCH_SIZE))

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

    # Training with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_center, \
        batch_angle_cls, batch_angle_res, batch_size_res = \
            get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx, NUM_CHANNEL)

        feed_dict = {ops["pointcloud_pl"]: batch_data,
                     ops["center_pl"]: batch_center,
                     ops["angle_cls_pl"]: batch_angle_cls,
                     ops["angle_res_pl"]: batch_angle_res,
                     ops["size_res_pl"]: batch_size_res,
                     ops["is_training_pl"]: is_training}

        ep = ops['end_points']
        summary, step, _, total_loss_val, center_loss_val, stage1_center_loss_val,\
        h_cls_loss_val, h_res_loss_val, s_res_loss_val, corners_loss_val, \
        iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['total_loss'],
                      ops['center_loss'], ops['stage1_center_loss'], ops['h_cls_loss'],
                      ops['h_res_loss'], ops['s_res_loss'], ops['corners_loss'],
                      ep['iou2ds'], ep['iou3ds']],
                     feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

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

        internal = 400
        if (batch_idx + 1) % internal == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean total loss: %f' % (total_loss_sum / internal))
            log_string('mean center loss: %f' % (center_loss_sum / internal))
            log_string('mean stage1 center loss: %f' % (stage1_center_loss_sum / internal))
            log_string('mean angle class loss: %f' % (h_cls_loss_sum / internal))
            log_string('mean angle res loss: %f' % (h_res_loss_sum / internal))
            log_string('mean size res loss: %f' % (s_res_loss_sum / internal))
            log_string('mean corners loss: %f' % (corners_loss_sum / internal))

            log_string('box IoU (ground/3D): %f / %f' % \
                (iou2ds_sum / float(BATCH_SIZE * internal), iou3ds_sum / float(BATCH_SIZE * internal)))
            log_string('box estimation accuracy (IoU=0.5): %f' % \
                       (float(iou3d_correct_cnt_50) / float(BATCH_SIZE * internal)))
            log_string('box estimation accuracy (IoU=0.7): %f' % \
                (float(iou3d_correct_cnt_70)/float(BATCH_SIZE * internal)))

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


def eval_one_epoch(sess, ops, val_writer):
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))
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
                     ops['is_training_pl']: is_training}

        ep = ops['end_points']
        summary, step, total_loss_val, center_loss_val, stage1_center_loss_val, h_cls_loss_val, \
        h_res_loss_val, s_res_loss_val, corners_loss_val, \
        iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'], ops['total_loss'],
                      ops['center_loss'], ops['stage1_center_loss'], ops['h_cls_loss'],
                      ops['h_res_loss'], ops['s_res_loss'], ops['corners_loss'],
                      ep['iou2ds'], ep['iou3ds']],
                     feed_dict=feed_dict)

        val_writer.add_summary(summary, step)

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

    EPOCH_CNT += 1


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
