# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings
import pdb
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Market(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "market1501"

    def __init__(self, root="", market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)
        self.pid_dict_test, self.cam_dict_test = self.build_dict(self.gallery_dir, is_train=True, mode = 'gallery')
        train = self.process_dir(self.train_dir, is_train=True, mode = 'train')
        query = self.process_dir(self.query_dir, is_train=True, mode = 'query')
        gallery = self.process_dir(self.gallery_dir, is_train=True, mode = 'gallery')
        super(Market, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, mode=''):
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

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if is_train:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            #camid -= 1  # index starts from 0
            if mode == 'train':
                pid = self.pid_dict[pid]
                camid = self.cam_dict[camid]     
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            else:   
                pid = self.pid_dict_test[pid]
                camid = self.cam_dict_test[camid]     
                pid += 751
                pid = int(pid)
                camid = int(camid)
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