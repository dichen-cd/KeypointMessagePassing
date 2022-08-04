import pickle
import os.path as osp
import numpy as np
import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch as TGeoBatch

from .serialization import mkdir_if_missing, write_json


class MyBatch(TGeoBatch):

    def cuda(self, **kwargs):
        self.batch = self.batch.cuda(**kwargs)
        self.x = self.x.cuda(**kwargs)
        # self.y = self.y.cuda(**kwargs)   # handled otherwisely
        self.edge_index = self.edge_index.cuda(**kwargs)
        return self

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = MyBatch()
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['batch']:
            batch[key] = []

        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value + cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Treat 0-dimensional tensors as 1-dimensional.
                if isinstance(item, Tensor) and item.dim() == 0:
                    item = item.unsqueeze(0)

                batch[key].append(item)

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                cat_dims[key] = cat_dim
                if isinstance(item, Tensor):
                    size = item.size(cat_dim)
                elif isinstance(item, SparseTensor):
                    size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f'{key}_{j}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long))
                    else:
                        tmp = f'{key}_batch'
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size, ), i, dtype=torch.long))

            if hasattr(data, '__num_nodes__'):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        # Fix initial slice values:
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()


keypoint_names = ['Nose', 'EyeL', 'EyeR', 'EarL', 'EarR',
                  'ShoulderL', 'ShoulderR', 'ElbowL', 'ElbowR', 'WristL', 'WristR',
                  'HipL', 'HipR', 'KneeL', 'KneeR', 'AnkleL', 'AnkleR']


def read_keypoints(path):
    """Reads keypoints from path.
    Args:
        path (str): path to an keypoints file.
    Returns:
        numpy array
    """
    got_kp = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_kp:
        try:
            with open(path, 'rb')as f:
                kps = f.readline()  # one-line only text
            kps = np.asarray(eval(kps), dtype=np.float32)
            kps[kps != kps] = 0.0  # detect nan and replace nan with 0
            got_kp = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return kps


def null_collate_fn(x):
    return x


def collate_torch_geo(x):
    '''
    x: list of dicts

        item = {'img': imgs,
                'keypoints': keypoints,
                'edge_index': edge_index,
                'pid': pid,
                'camid': camid}

    '''
    out_list = []
    imgs = []
    num_frames = []
    camids = []
    pids = []
    for item in x:
        out_list.append(
            Data(x=item['keypoints'],
                 edge_index=item['edge_index'].t())
        )
        imgs.append(item['img'])
        num_frames.append(len(item['img']))
        camids.append(item['camid'])
        pids.append(item['pid'])

    return {
        'num_frames': num_frames,
        'data': MyBatch.from_data_list(out_list),
        'img' : torch.stack(imgs, dim=0) if imgs[0][0] is not None else None,
        'camid': camids,
        'pid': torch.tensor(pids, dtype=torch.long)
    }


def visualize_ranked_results(
    distmat, dataset, data_type, width=128, height=256, save_dir='', topk=10
):
    """
    adapted from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/reidtools.py
    
    Visualizes ranked results.
    Supports both image-reid and video-reid.
    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid, dsetid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    # mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)

    result = {}

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(
                    dst, prefix + '_top' + str(rank).zfill(3)
                ) + '_' + osp.basename(src[0]) + '_' + suffix
            else:
                # dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
                dst = osp.basename(src[0])
            # mkdir_if_missing(dst)
            # for img_path in src:
                # shutil.copy(img_path, dst)
            return dst
        else:
            dst = osp.join(
                dst, prefix + '_top' + str(rank).zfill(3) + '_name_' +
                osp.basename(src)
            )
            # shutil.copy(src, dst)
            return dst

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx][:3]
        qimg_path_name = qimg_path[0] if isinstance(
            qimg_path, (tuple, list)
        ) else qimg_path

        qdir = osp.join(
            save_dir, osp.basename(osp.splitext(qimg_path_name)[0])
        )
        # mkdir_if_missing(qdir)
        query_key = _cp_img_to(qimg_path, '', rank=0, prefix='query')

        result[query_key] = []

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx][:3]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid == qpid
                
                if matched and rank_idx == 1:
                    print(f'Matched! query: {qimg_path[0].split("/")[-1]}, top-1: {gimg_path[0].split("/")[-1]}')
                # if rank_idx == 1 and not matched:
                #     print(f'UnMatched! query: {qimg_path[0].split("/")[-1]}, top-1: {gimg_path[0].split("/")[-1]}')
                
                gallery_value = _cp_img_to(
                        gimg_path,
                        '',
                        rank=rank_idx,
                        prefix='gallery',
                        matched=matched
                    )
                result[query_key].append(gallery_value)

                rank_idx += 1
                if rank_idx > topk:
                    break

        # if data_type == 'image':
        #     imname = osp.basename(osp.splitext(qimg_path_name)[0])
        #     cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    write_json(result, save_dir + '.json')
    print('Done. Re-ID results have been saved to "{}" ...'.format(save_dir + '.json'))

