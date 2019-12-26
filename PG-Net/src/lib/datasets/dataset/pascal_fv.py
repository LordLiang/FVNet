from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import os

import torch.utils.data as data

class PascalFV(data.Dataset):
  num_classes = 3
  default_resolution = [128, 512]
  mean = np.array([0.221, 0.173, 0.411], 
                   dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.161, 0.150, 0.223], 
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(PascalFV, self).__init__()
    self.split = split
    if split in ['train', 'val']:
        self.folder = 'training'
    else:
        self.folder = 'testing'
    self.data_dir = os.path.join(opt.data_dir, 'kitti_fvnet', 'projection', self.folder)
    self.img_dir = os.path.join(self.data_dir, 'images')
    _ann_name = {'train': 'train', 'val': 'val'}
    self.annot_path = os.path.join(
      self.data_dir, 'annotations', 
      'kitti_{}.json').format(_ann_name[split])
    self.max_objs = 50
    self.class_name = ['__background__','Car', 'Pedestrian', 'Cyclist']
    self._valid_ids = np.arange(1, 4, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing pascal {} data.'.format(_ann_name[split]))
    self.coco = coco.COCO(self.annot_path)
    self.images = sorted(self.coco.getImgIds())
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = [[[] for __ in range(self.num_samples)] \
                  for _ in range(self.num_classes + 1)]
    for i in range(self.num_samples):
      img_id = self.images[i]
      for j in range(1, self.num_classes + 1):
        if isinstance(all_bboxes[img_id][j], np.ndarray):
          detections[j][i] = all_bboxes[img_id][j].tolist()
        else:
          detections[j][i] = all_bboxes[img_id][j]
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/{}_results.json'.format(save_dir, self.split), 'w'))

  def run_eval(self, results, save_dir):
    os.system('python tools/reval_fv.py ' + \
              '{}/{}_results.json'.format(save_dir, self.split) + \
              ' --test_split {}'.format(self.split))
