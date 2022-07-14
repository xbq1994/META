import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import pdb
__all__ = ['CUHK02', ]

@DATASET_REGISTRY.register()
class CUHK02(ImageDataset):
    """CUHK02.
    Reference:
        Li and Wang. Locally Aligned Feature Transforms across Views. CVPR 2013.
    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>`_
    
    Dataset statistics:
        - 5 camera view pairs each with two cameras
        - 971, 306, 107, 193 and 239 identities from P1 - P5
        - totally 1,816 identities
        - image format is png
    Protocol: Use P1 - P4 for training and P5 for evaluation.
    Note: CUHK01 and CUHK02 overlap.
    """
    dataset_dir = 'cuhk02'
    cam_pairs = ['P1', 'P2', 'P3', 'P4', 'P5']
    test_cam_pair = 'P5'
    dataset_name = "cuhk02"

    def __init__(self, root='', **kwargs):
        self.root = "/export/DATA/"
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)
        self.train_pids=set()
        self.query_pids=set()
        self.gallery_pids=set()
        train, query, gallery = self.get_data_list()
        super(CUHK02, self).__init__(train, query, gallery, **kwargs)
        
    def get_data_list(self):
        num_train_pids, camid = 0, 0
        train, query, gallery = [], [], []

        for cam_pair in self.cam_pairs:
            cam_pair_dir = osp.join(self.dataset_dir, cam_pair)

            cam1_dir = osp.join(cam_pair_dir, 'cam1')
            cam2_dir = osp.join(cam_pair_dir, 'cam2')

            impaths1 = glob.glob(osp.join(cam1_dir, '*.png'))
            impaths2 = glob.glob(osp.join(cam2_dir, '*.png'))

            if cam_pair == self.test_cam_pair:
                # add images to query
                for impath in impaths1:
                    pid = int(osp.basename(impath).split('_')[0])+1577
                    self.query_pids.add(pid)
                    pid = self.dataset_name + "_" + str(pid)
                    query.append((impath, pid, self.dataset_name + "_" + str(camid)))
                camid += 1

                # add images to gallery
                for impath in impaths2:
                    pid = int(osp.basename(impath).split('_')[0])+1577
                    pid = int(pid)
                    self.gallery_pids.add(pid)
                    pid = self.dataset_name + "_" + str(pid)
                    gallery.append((impath, pid, self.dataset_name + "_" + str(camid)))
                camid += 1

            else:
                pids1 = [
                    osp.basename(impath).split('_')[0] for impath in impaths1
                ]
                pids2 = [
                    osp.basename(impath).split('_')[0] for impath in impaths2
                ]
                pids = set(pids1 + pids2)
                pid2label = {
                    pid: label + num_train_pids
                    for label, pid in enumerate(pids)
                }

                # add images to train from cam1
                for impath in impaths1:
                    pid = osp.basename(impath).split('_')[0]
                    self.train_pids.add(pid2label[pid])
                    pid = self.dataset_name + "_" + str(pid2label[pid])
                    train.append((impath, pid, self.dataset_name + "_" + str(camid)))
                camid += 1

                # add images to train from cam2
                for impath in impaths2:
                    pid = osp.basename(impath).split('_')[0]
                    self.train_pids.add(pid2label[pid])
                    pid = self.dataset_name + "_" + str(pid2label[pid])
                    train.append((impath, pid, self.dataset_name + "_" + str(camid)))
                camid += 1
                num_train_pids += len(pids)

        return train, query, gallery