import logging as log
import torch
from torchvision import transforms as tf
from statistics import mean
import os
from PIL import Image

from .. import data as vn_data
from .. import models
from . import engine
from utils.test import voc_wrapper


__all__ = ['VOCTestingEngine']

ROOT = "/adata/zhoujie/yolo-kitti4"

TESTSET = [
    ('testing', 'test'),
    ]


class TestDataset(vn_data.Dataset):
    def __init__(self, hyper_params):
        network_size = hyper_params.network_size
        lb  = vn_data.transform.Letterbox(network_size)
        it  = tf.ToTensor()
        self.img_tf = vn_data.transform.Compose([lb, it])

        self.keys = []
        for (tag, img_set) in TESTSET:
            with open(f'{ROOT}/{tag}/ImageSets/{img_set}.txt', 'r') as f:
                ids = f.read().strip().split()
            self.keys += [f'{ROOT}/{tag}/PNGImages/{idx}.png' for idx in ids]

        super(TestDataset, self).__init__(network_size)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        img = Image.open(self.keys[index % len(self)])

        # transform
        if self.img_tf is not None:
            img = self.img_tf(img)
        anno = []
        return img, anno


def VOCTestingEngine(hyper_params):
    log.debug('Creating network')

    model_name = hyper_params.model_name
    batch = hyper_params.batch
    use_cuda = hyper_params.cuda
    weights = hyper_params.weights
    conf_thresh = hyper_params.conf_thresh
    network_size = hyper_params.network_size
    labels = hyper_params.labels
    nworkers = hyper_params.nworkers
    pin_mem = hyper_params.pin_mem
    nms_thresh = hyper_params.nms_thresh
    #prefix = hyper_params.prefix
    results = hyper_params.results

    test_args = {'conf_thresh': conf_thresh, 'network_size': network_size,  'labels': labels}
    net = models.__dict__[model_name](hyper_params.classes, weights, train_flag=2, test_args=test_args)
    net.eval()
    log.info('Net structure\n%s' % net)
    #import pdb
    #pdb.set_trace()
    if use_cuda:
        net.cuda()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        TestDataset(hyper_params),
        batch_size = batch,
        shuffle = False,
        drop_last = False,
        num_workers = nworkers if use_cuda else 0,
        pin_memory = pin_mem if use_cuda else False,
        collate_fn = vn_data.list_collate,
    )

    log.debug('Running network')
    det = {}

    for idx, (data, box) in enumerate(loader):

        if (idx + 1) % 20 == 0:
            log.info('%d/%d' % (idx + 1, len(loader)))
        if use_cuda:
            data = data.cuda()
        with torch.no_grad():
            output, _= net(data)

        key_val = len(det)
        det.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(output)})

    netw, neth = network_size
    reorg_dets = voc_wrapper.reorgDetection(det, netw, neth) #, prefix)
    print(results)
    voc_wrapper.genResults(reorg_dets, results, nms_thresh)

