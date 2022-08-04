from __future__ import division, print_function, absolute_import
import os.path as osp
import warnings

from ..dataset import PoseDataset
from ....utils.serialization import read_json


class PoseTrackReID(PoseDataset):
    '''
    Dataset statistics:

    '''
    dataset_dir = 'posetrack-reid'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.train_name_path = osp.join(
            self.dataset_dir, 'train_names.json'
        )
        self.query_name_path = osp.join(
            self.dataset_dir, 'query_names.json'
        )
        self.gallery_name_path = osp.join(
            self.dataset_dir, 'gallery_names.json'
        )

        required_files = [
            self.train_name_path,
            self.query_name_path, self.gallery_name_path
        ]
        self.check_before_run(required_files)

        train = self.process_data(
            read_json(self.train_name_path),
            home_dir='train_keypoints', relabel=True
        )
        query = self.process_data(
            read_json(self.query_name_path),
            home_dir='val_keypoints', relabel=False
        )
        gallery = self.process_data(
            read_json(self.gallery_name_path),
            home_dir='val_keypoints', relabel=False
        )

        super(PoseTrackReID, self).__init__(train, query, gallery, **kwargs)

    def process_data(self, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['train_keypoints', 'val_keypoints']
        num_tracklets = len(meta_data)
        pids = set()
        if relabel:
            for items in meta_data:
                pid = items[1]
                pids.add(pid)
            pids = list(pids)
            pid2label = {pid: label for label, pid in enumerate(pids)}

        tracklets = []
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx]
            img_names, orig_pid, vid = data
            if relabel:
                pid = pid2label[orig_pid]
            else:
                pid = orig_pid

            keypoints_paths = [
                osp.join(self.dataset_dir, home_dir, str(orig_pid), img_name)
                for img_name in img_names
            ]

            if len(keypoints_paths) >= min_seq_len:
                keypoints_paths = tuple(keypoints_paths)
                tracklets.append((keypoints_paths, pid, vid))

        return tracklets