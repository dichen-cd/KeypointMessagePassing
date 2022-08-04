from __future__ import division, print_function, absolute_import

import copy
import numpy as np
import torch
from torchreid.data.datasets import VideoDataset
from torchreid.utils import read_image

from ...utils.tools import read_keypoints


'''
class Joints:              class JointsHeadless:
    Nose = 0
    Leye = 1
    Reye = 2
    LEar = 3
    REar = 4
    LS = 5                     LS = 0
    RS = 6                     RS = 1
    LE = 7                     LE = 2
    RE = 8                     RE = 3
    LW = 9                     LW = 4
    RW = 10                    RW = 5
    LH = 11                    LH = 6
    RH = 12                    RH = 7
    LK = 13                    LK = 8
    RK = 14                    RK = 9
    LA = 15                    LA = 10
    RA = 16                    RA = 11

'''

link_pairs = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6],
    [6, 8], [8, 10],
    # reverse connection
    [13, 15], [11, 13], [5, 11],
    [14, 12], [16, 14], [6, 12],
    [1, 3], [2, 1], [0, 1], [2, 0], [4, 2],
    [7, 9], [5, 7], [6, 5],
    [8, 6], [10, 8]
]


link_pairs_headless = [
    [10, 8], [8, 6], [6, 0],
    [7, 9], [9, 11], [7, 1],
    [4, 2], [2, 0], [0, 1],
    [1, 3], [3, 5],
    # reverse connection
    [8, 10], [6, 8], [0, 6],
    [9, 7], [11, 9], [1, 7],
    [2, 4], [0, 2], [1, 0],
    [3, 1], [5, 3]
]


oks_sigmas = torch.Tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                           1.07, 1.07, .87, .87, .89, .89]) / 10.0


def normalize_pose(pf, keep_orig_coord=False,
                   orig_img_width=None, orig_img_height=None, epsilon=2.0):
    if keep_orig_coord:
        orig_coord = pf.copy()[:, :2]
        orig_coord[:, 0] /= orig_img_width
        orig_coord[:, 1] /= orig_img_height

    lx, ly = np.min(pf[:, 0]), np.min(pf[:, 1])
    pf[:, 0] -= lx
    pf[:, 1] -= ly
    pose_width = np.max(pf[:, 0])
    pose_height = np.max(pf[:, 1])
    if pose_width == 0:
        pose_width = epsilon
    if pose_height == 0:
        pose_height = epsilon    
    ct_x = np.mean(pf[:, 0])
    ct_y = np.mean(pf[:, 1])
    pf[:, 0] = (pf[:, 0] - ct_x) / (pose_width / 2.0)
    pf[:, 1] = (pf[:, 1] - ct_y) / (pose_height / 2.0)
    if keep_orig_coord:
        return np.concatenate([orig_coord, pf], axis=1)
    else:
        return pf


def select_window(index, length, window_size=5):
    if index <= window_size // 2:
        return list(range(window_size))
    elif index >= length - window_size // 2:
        return list(range(length - window_size, length))
    else:
        return list(range(index - window_size // 2, index + 1 + window_size // 2))


def oks_iou(pose_i, pose_j, scale=1.0, sigmas=oks_sigmas):
    '''
    pose_i: torch.Tensor, size = (17 or 12, 3 or 5)
    pose_j: same to pose_i
    '''
    if pose_i.shape[0] == 12:
        sigmas = sigmas[5:]

    if pose_i.shape[1] == 5:
        dx = pose_i[:, 2] - pose_j[:, 2]  # do it in the normalized pose space
        dy = pose_i[:, 3] - pose_j[:, 3]
    elif pose_i.shape[1] == 3:
        dx = pose_i[:, 0] - pose_j[:, 0]
        dy = pose_i[:, 1] - pose_j[:, 1]
    e = (dx**2 + dy**2) / (2 * scale**2 * sigmas**2)
    return torch.sum(torch.exp(-e)) / pose_i.shape[0]


def filter_glitches(keypoints, thresh=0.2, window_size=5):
    '''
    keypoints: torch.Tensor, size = (num_frames, num_keypoints, 3)
    '''
    num_frames = keypoints.shape[0]
    for i in range(num_frames):
        window_index = select_window(i, num_frames, window_size)
        
        s = []
        for j in window_index:
            if j != i:
                s.append(oks_iou(keypoints[i], keypoints[j]))

        if np.mean(s) < thresh:  # frame i is a glitch
            if i == 0:
                keypoints[i] = keypoints[i + 1]
            elif i == num_frames - 1:
                keypoints[i] = keypoints[i - 1]
            else:
                keypoints[i] = (keypoints[i - 1] * 0.8 + keypoints[i + 1] * 0.2)  
                # previous frames are more reliable since it's already fixed.
    return keypoints


def append_dynamics_features(keypoints, velocity=True, acceleration=True):
    '''
    keypoints: torch.Tensor, size = (num_frames, num_keypoints, 3 or 5)
    '''
    if keypoints.shape[2] == 3:
        position = keypoints[:, :, :2]  # size = (num_frames, num_keypoints, 2)
        confidence = keypoints[:, :, 2:]  # size = (num_frames, num_keypoints, 1)
    elif keypoints.shape[2] == 5:
        orig_position = keypoints[:, :, :2]
        position = keypoints[:, :, 2:4]  # size = (num_frames, num_keypoints, 2)
        confidence = keypoints[:, :, 4:]  # size = (num_frames, num_keypoints, 1)
    
    out = position

    if velocity:
        v = position[1:] - position[:-1]  # size = (num_frames-1, num_keypoints, 2)
        v = torch.cat([v, v[-1:]], dim=0)  # simply repeat the velocity of the last frame
        out = torch.cat([out, v], dim=2)

    if acceleration:
        a = v[1:] - v[:-1]
        a = torch.cat([a, a[-1:]], dim=0)
        out = torch.cat([out, a], dim=2)

    if keypoints.shape[2] == 3:
        out = torch.cat([out, confidence], dim=2)
    elif keypoints.shape[2] == 5:
        out = torch.cat([orig_position, out, confidence], dim=2)

    return out


class PoseDataset(VideoDataset):

    def __init__(
        self,
        train,
        query,
        gallery,
        seq_len=15,
        sample_method='evenly',
        return_img=False,
        return_pose_graph=True,
        headless=False,
        rm_glitches=False,
        include_dynamics=False,
        include_spatial_links=True,
        include_temporal_links=True,
        normalize_confidence=True,
        **kwargs
    ):
        super(PoseDataset, self).__init__(
            train, query, gallery,
            seq_len=seq_len, sample_method=sample_method, **kwargs)

        if self.transform is None:
            raise RuntimeError('transform cannot be None')

        self.return_img = return_img
        self.return_pose_graph = return_pose_graph
        self.headless = headless
        self.rm_glitches = rm_glitches
        self.include_dynamics = include_dynamics
        self.include_spatial_links = include_spatial_links
        self.include_temporal_links = include_temporal_links
        self.keep_orig_coord = self.return_img
        self.normalize_confidence = normalize_confidence

    def __getitem__(self, idx):
        # img_paths, pose_paths, pid, camid = self.data[idx]
        # num_imgs = len(img_paths)
        pose_paths, pid, camid = self.data[idx]
        num_imgs = len(pose_paths)
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
            keypoints_path = pose_paths[int(index)]
            if self.return_img:
                # img_path = img_paths[int(index)]
                img_path = keypoints_path.replace('.pose', '.jpg')
                img_path = img_path.replace('_keypoints', '')
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

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pose_path(s), pid, camid)
            update: data (list): contains tuples of (pose_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        # for _, _, pid, camid in data:
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)

    def __add__(self, other):
        raise NotImplementedError

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        # for _, _, pid, _ in self.gallery:
        for _, pid, _ in self.gallery:
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            # for img_path, pose_path, pid, camid in data:
            for pose_path, pid, camid in data:
                if pid in self._junk_pids:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                # combined.append((img_path, pose_path, pid, camid))
                combined.append((pose_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    @staticmethod
    def graph_gen(keypoints, headless=False, spatial=True, temporal=True):
        N, num_kps, dim = keypoints.shape

        if headless:
           base_pairs = link_pairs_headless
        else:
           base_pairs = link_pairs

        base_skelenton = torch.tensor(base_pairs, dtype=torch.long)

        edge_index = []
        if spatial:
            # intra-frame edges
            for i in range(N):
                edge_index.append(base_skelenton + i * num_kps)

        if temporal:
            # inter_frame edges
            for i in range(N - 1):  # N - 1: the last frame has no connections to the next
                links = [[i * num_kps + j, (i + 1) * num_kps + j]
                         for j in range(num_kps)]
                edge_index.append(
                    torch.tensor(links, dtype=torch.long)
                )
                reverse_links = [[(i + 1) * num_kps + j, i * num_kps + j]
                                 for j in range(num_kps)]
                edge_index.append(
                    torch.tensor(reverse_links, dtype=torch.long)
                )

        return keypoints.reshape(N * num_kps, dim), torch.cat(edge_index, dim=0)
