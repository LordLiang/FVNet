from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset
from .sample.fvdet import FVDetDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.pascal_fv import PascalFV
from .dataset.pascal_fv2 import PascalFV2

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'fv': PascalFV,
  'fv2': PascalFV2
}

_sample_factory = {
  'ctdet': CTDetDataset,
  'fvdet': FVDetDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
