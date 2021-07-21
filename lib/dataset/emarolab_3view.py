# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pickle
import collections

from dataset.joints_dataset import JointsDataset
import multiviews.cameras as cam_utils


class Emarolab3View(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rhee',
            4: 'rfoo',
            5: 'rtoe',
            6: 'lhip',
            7: 'lkne',
            8: 'lhee',
            9: 'lfoo',
            10: 'ltoe', 
            11: 'belly',
            12: 'neck',
            13: 'nose',
            14: 'head',
            15: 'lsho',
            16: 'lelb',
            17: 'lwri',
            18: 'rsho',
            19: 'relb',
            20: 'rwri'
        }

        self.u2a_mapping = super().get_mapping()

        grouping_db_pickle_file = osp.join(self.root,
                                           'annotations_quickload.pkl')
        if osp.isfile(grouping_db_pickle_file):
            with open(grouping_db_pickle_file, 'rb') as f:
                grouping_db = pickle.load(f)
                self.grouping = grouping_db['grouping']
                self.db = grouping_db['db']
        else:
            anno_file = osp.join(self.root, 'annotations.pkl')
            self.db = self.load_db(anno_file)

            self.u2a_mapping = super().get_mapping()
            super().do_mapping()

            self.grouping = self.get_group(self.db)
            grouping_db_to_dump = {'grouping': self.grouping, 'db': self.db}
            with open(grouping_db_pickle_file, 'wb') as f:
                pickle.dump(grouping_db_to_dump, f)

        #Filter for only one walking_scene
        #self.grouping = self.groups_query(self.db, self.grouping, "Walk", 1, 9, True)


        if self.is_train:
            self.grouping = self.grouping[::20]
        else:
            self.grouping = self.grouping[::64]
            #pass


        self.group_size = len(self.grouping)
        self.selected_cam = [0, 1, 2]

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1]
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        return filtered_grouping

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item)
            # data type convert to float32
            m['scale'] = m['scale'].astype(np.float32)
            m['center'] = m['center'].astype(np.float32)
            m['rotation'] = int(m['rotation'])
            if 'name' in m['camera']:
                del m['camera']['name']
            for k in m['camera']:
                m['camera'][k] = m['camera'][k].astype(np.float32)

            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
        return 's_{:02}_act_{}_timerange_{}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['timerange'],
            datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()
        nview = 4
        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            threshold = 0.0125
        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        flat_items = []
        box_lengthes = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
                flat_items.append(self.db[item])
                # crop img pixels
                boxsize = np.array(self.db[item]['scale']).sum() * 100.0
                box_lengthes.append(boxsize)
        gt = np.array(gt)
        if pred.shape[1] == 20 or pred.shape[1] == 24:
            pred = pred[:, su, :2]
        elif pred.shape[1] == 17:
            pred = pred[:, :, :2]
        detection_threshold = np.array(
            box_lengthes).reshape((-1, 1)) * threshold

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= detection_threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        detected_int = detected.astype(np.int)
        nsamples, njoints = detected.shape
        per_grouping_detected = detected_int.reshape(
            nsamples // nview, nview * njoints)
        return name_values, np.mean(joint_detection_rate), per_grouping_detected

    def evaluate_3d(self, preds3d, thresholds=None):
        if thresholds is None:
            thresholds = [5., 10., 15., 20., 25., 50., 75., 100., 125., 150., ]

        gt3d = []
        for idx, items in enumerate(self.grouping):
            # note that h36m joints_3d is in camera frame
            db_rec = self.db[items[0]]
            j3d_global = cam_utils.camera_to_world_frame(
                db_rec['joints_3d_camera'], db_rec['camera']['R'], db_rec['camera']['T'])
            gt3d.append(j3d_global)
        gt3d = np.array(gt3d)

        assert preds3d.shape == gt3d.shape, 'shape mismatch of preds and gt'
        distance = np.sum((preds3d - gt3d)**2, axis=2)

        num_groupings = len(gt3d)
        pcks = []
        for thr in thresholds:
            detections = distance <= thr**2
            detections_perjoint = np.sum(detections, axis=0)
            pck_perjoint = detections_perjoint / num_groupings
            # pck_avg = np.average(pck_perjoint, axis=0)
            pcks.append(pck_perjoint)

        return thresholds, pcks
