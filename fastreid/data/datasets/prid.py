# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import sys
import os
import os.path as osp
import random
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
import json
import time
import errno
import numpy as np
import warnings
import PIL
import torch
from PIL import Image
import pdb
__all__ = ['PRID', ]

@DATASET_REGISTRY.register()
class PRID(ImageDataset):
    """PRID (single-shot version of prid-2011)
    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and Discriminative
        Classification. SCIA 2011.
    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_
    
    Dataset statistics:
        - Two views.
        - View A captures 385 identities.
        - View B captures 749 identities.
        - 200 identities appear in both views.
    """
    dataset_dir = 'prid2011'
    dataset_url = None

    def __init__(self, root='', split_id=9, **kwargs):
        self.root = "/export/DATA/"
        self.dataset_dir = osp.join(self.root)

        self.cam_a_dir = osp.join(self.dataset_dir, 'single_shot', 'cam_a')
        self.cam_b_dir = osp.join(self.dataset_dir, 'single_shot', 'cam_b')
        self.split_path = osp.join(self.dataset_dir, 'splits_single_shot.json')

        required_files = [
            self.dataset_dir,
            self.cam_a_dir,
            self.cam_b_dir
        ]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError('split_id exceeds range, received {}, but expected between 0 and {}'.format(split_id, len(splits)-1))
        split = splits[split_id]

        train, query, gallery = self.process_split(split)
        super().__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating splits ...')

            splits = []
            for _ in range(10):
                # randomly sample 100 IDs for train and use the rest 100 IDs for test
                # (note: there are only 200 IDs appearing in both views)
                pids = [i for i in range(1, 201)]
                train_pids = random.sample(pids, 100)
                train_pids.sort()
                test_pids = [i for i in pids if i not in train_pids]
                split = {'train': train_pids, 'test': test_pids}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file is saved to {}'.format(self.split_path))

    def process_split(self, split):
        train, query, gallery = [], [], []
        train_pids = split['train']
        test_pids = split['test']

        train_pid2label = {pid: label for label, pid in enumerate(train_pids)}

        # train
        train = []
        for pid in train_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            pid = train_pid2label[pid]
            img_a_path = osp.join(self.cam_a_dir, img_name)
            train.append((img_a_path, pid, 0))
            img_b_path = osp.join(self.cam_b_dir, img_name)
            train.append((img_b_path, pid, 1))

        # query and gallery
        query, gallery = [], []
        for pid in test_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_a_path = osp.join(self.cam_a_dir, img_name)
            query.append([img_a_path, pid, 0])
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
        for pid in range(201, 750):
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
        
        return train, query, gallery
        
def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))