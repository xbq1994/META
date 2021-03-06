# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import pdb

@DATASET_REGISTRY.register()
class DukeMTMC(ImageDataset):
    """DukeMTMC-reID.

    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.

    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_

    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'DukeMTMC-reID'
    dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
    dataset_name = "dukemtmc"

    def __init__(self, root="/export/home/lg/xbq/Data/", **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = "/export/DATA/"
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)
        self.pid_dict_test, self.cam_dict_test = self.build_dict(self.gallery_dir, is_train=True, mode = 'gallery')
        
        train = self.process_dir(self.train_dir, is_train=True, mode='train')
        query = self.process_dir(self.query_dir, is_train=True, mode='query')
        gallery = self.process_dir(self.gallery_dir, is_train=True, mode='gallery')

        super(DukeMTMC, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, mode=''):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        pid_set = set()
        cam_set = set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            pid_set.add(pid)
            cam_set.add(camid)

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if is_train:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            #camid -= 1  # index starts from 0            
            if mode == 'train':
                pid = self.pid_dict[pid]
                camid = self.cam_dict[camid]     
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            else:   
                pid = self.pid_dict_test[pid]
                camid = self.cam_dict_test[camid]     
                pid += 702
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))
        return data

    def build_dict(self, dir_path, is_train=True, mode=''):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        pid_set = set()
        cam_set = set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_set.add(pid)
            cam_set.add(camid)
        
        pids = sorted(list(pid_set))
        cams = sorted(list(cam_set))

        pid_dict = dict([(p, i) for i, p in enumerate(pids)])
        cam_dict = dict([(p, i) for i, p in enumerate(cams)])

        return pid_dict, cam_dict