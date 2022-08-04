from __future__ import division, print_function, absolute_import
import os.path as osp
import warnings

from torchreid.data.datasets import VideoDataset
from ....utils.serialization import read_json


class PoseTrackReIDVO(VideoDataset):
    '''
    VO: visual only

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
            home_dir='train', relabel=True
        )
        query = self.process_data(
            read_json(self.query_name_path),
            home_dir='val', relabel=False
        )
        gallery = self.process_data(
            read_json(self.gallery_name_path),
            home_dir='val', relabel=False
        )

        super(PoseTrackReIDVO, self).__init__(train, query, gallery, **kwargs)

    def process_data(self, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['train', 'val']
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

            frame_paths = [
                osp.join(self.dataset_dir, home_dir,
                         str(orig_pid), img_name.replace('.pose', '.jpg'))
                for img_name in img_names
            ]

            if len(frame_paths) >= min_seq_len:
                frame_paths = tuple(frame_paths)
                tracklets.append((frame_paths, pid, vid))

        return tracklets
