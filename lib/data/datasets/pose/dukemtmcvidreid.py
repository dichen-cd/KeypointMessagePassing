from __future__ import division, print_function, absolute_import
import os.path as osp
import warnings
from scipy.io import loadmat
import numpy as np

from torchreid.utils import read_image

from ..dataset import *
from ....utils.serialization import read_json
from ....utils.tools import read_keypoints


class DukeMTMCVidReIDPose(PoseDataset):
    """DukeMTMCVidReID.
    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target,
          Multi-Camera Tracking. ECCVW 2016.
        - Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
          Re-Identification by Stepwise Learning. CVPR 2018.
    URL: `<https://github.com/Yu-Wu/DukeMTMC-VideoReID>`_
    
    Dataset statistics:
        - identities: 702 (train) + 702 (test).
        - tracklets: 2196 (train) + 2636 (test).
    """
    dataset_dir = 'dukemtmc-vidreid'
    dataset_url = None

    def __init__(self, root='', min_seq_len=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.train_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/train')
        self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/query')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'DukeMTMC-VideoReID/gallery'
        )
        self.split_train_json_path = osp.join(
            self.dataset_dir, 'split_train.json'
        )
        self.split_query_json_path = osp.join(
            self.dataset_dir, 'split_query.json'
        )
        self.split_gallery_json_path = osp.join(
            self.dataset_dir, 'split_gallery.json'
        )
        self.min_seq_len = min_seq_len

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(
            self.train_dir, self.split_train_json_path, relabel=True
        )
        query = self.process_dir(
            self.query_dir, self.split_query_json_path, relabel=False
        )
        gallery = self.process_dir(
            self.gallery_dir, self.split_gallery_json_path, relabel=False
        )

        super(DukeMTMCVidReIDPose, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            split = read_json(json_path)
            return split['tracklets']

        print('=> Generating split json file (** this might take a while **)')
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print(
            'Processing "{}" with {} person identities'.format(
                dir_path, len(pdirs)
            )
        )

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        tracklets = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel:
                pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx + 1).zfill(4)
                    res = glob.glob(
                        osp.join(tdir, '*' + img_idx_name + '*.jpg')
                    )
                    if len(res) == 0:
                        warnings.warn(
                            'Index name {} in {} is missing, skip'.format(
                                img_idx_name, tdir
                            )
                        )
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        print('Saving split to {}'.format(json_path))
        split_dict = {'tracklets': tracklets}
        write_json(split_dict, json_path)

        return tracklets

    def __getitem__(self, idx):
        # img_paths, pid, camid = self.data[idx]
        img_paths, pid, camid, dsetid = self.data[idx]
        num_imgs = len(img_paths)
        sample_seq_len = self.seq_len + 2  # over-sample 2 imgs for velocity and acceleration

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= sample_seq_len else True
            indices = np.random.choice(
                indices, size=sample_seq_len, replace=replace
            )
            # sort indices to keep temporal order (comment it to be
            # order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= sample_seq_len:
                num_imgs -= num_imgs % sample_seq_len
                indices = np.arange(0, num_imgs, num_imgs / sample_seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = sample_seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs - 1)
                    ]
                )
            assert len(indices) == sample_seq_len

        elif self.sample_method == 'conseq':
            # Samples concecutive seq_len images from a tracklet
            if num_imgs >= sample_seq_len:
                startpoint = np.random.randint(
                    low=0, high=num_imgs - sample_seq_len + 1)
                indices = np.arange(startpoint, startpoint + sample_seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = sample_seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs - 1)
                    ]
                )
            assert len(indices) == sample_seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)

        else:
            raise ValueError(
                'Unknown sample method: {}'.format(self.sample_method)
            )

        imgs, keypoints = [], []
        for index in indices:
            img_path = img_paths[int(index)]
            keypoints_path = img_path.replace('.jpg', '.pose')
            if self.return_img:
                img = read_image(img_path)
                orig_img_width, orig_img_height = img.size
            else:
                img = None
                orig_img_width, orig_img_height = None, None

            kps = read_keypoints(keypoints_path)  # numpy array, size = (17, 3)
            if self.headless:
                kps = kps[5:]  # numpy array, size = (12, 3)

            if self.transform is not None:
                # img, kps = self.transform(img, kps)  # TODO!!!! re-write all
                # trandform functions
                if self.return_img:
                    img = self.transform(img)
                kps = normalize_pose(kps, self.keep_orig_coord, orig_img_width, orig_img_height)
                kps = torch.from_numpy(kps)
            if self.return_img:
                img = img.unsqueeze(0)  # img must be torch.Tensor

            kps = kps.unsqueeze(0)  # torch.Tensor, size = (1, 17 or 12, 3 or 5)
            imgs.append(img)
            keypoints.append(kps)

        if self.return_img:
            imgs = torch.cat(imgs, dim=0)
        # torch.Tensor, size = (N, 17 or 12, 3)
        keypoints = torch.cat(keypoints, dim=0)
        if self.rm_glitches:
            keypoints = filter_glitches(keypoints)

        if self.include_dynamics:
            keypoints = append_dynamics_features(keypoints)

        keypoints = keypoints[:self.seq_len]  # remove the over-sampled imgs
        imgs = imgs[:self.seq_len]

        if self.return_pose_graph:
            keypoints, edge_index = self.graph_gen(keypoints, self.headless,
                                                   self.include_spatial_links,
                                                   self.include_temporal_links)

        if self.normalize_confidence:
            keypoints[:, -1] = keypoints[:, -1] - keypoints[:, -1].mean() + 1

        item = {'img': imgs,
                'keypoints': keypoints,
                'edge_index': edge_index,
                'pid': pid,
                'camid': camid,
                'img_path': img_path}

        return item
