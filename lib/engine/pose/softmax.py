from __future__ import division, print_function, absolute_import
import numpy as np
import torch
# from torch.cuda.amp import autocast, GradScaler
from torchreid import metrics

from ..video import VideoSoftmaxEngine


class PoseSoftmaxEngine(VideoSoftmaxEngine):

    def parse_data_for_train(self, data):
        inputs = data['data']
        num_frames = data['num_frames']
        # b: batch size
        # s: sqeuence length
        # c: channel depth, equals 17
        # d: dim, equals 3 (x, y, score)
        b = inputs.num_graphs
        s = num_frames[0]
        # assert np.all([nf == s for nf in num_frames])
        # c, d = 17, 3
        pids = data['pid']
        pids = pids.view(b, 1).expand(b, s)
        pids = pids.contiguous().view(b * s)
        return inputs, pids

    def parse_data_for_eval(self, data):
        inputs = data['data']
        pids = data['pid']
        camids = data['camid']
        return inputs, pids, camids

    def extract_features(self, inputs):
        # b: batch size
        # s: sqeuence length
        # c: channel depth
        # h: height
        # w: width
        b = inputs.num_graphs
        features = self.model(inputs)
        feature_dim = features.shape[-1]
        features = features.view(b, -1, feature_dim)
        if self.pooling_method == 'avg':
            features = torch.mean(features, 1)
        else:
            features = torch.max(features, 1)[0]
        return features


class PoseWVideoSoftmaxEngine(PoseSoftmaxEngine):
    
    # def __init__(self, *args, **kwargs):
    #     super(VideoSoftmaxEngine, self).__init__(*args, **kwargs)
    #     self.scaler = GradScaler()

    def parse_data_for_train(self, data):
        graphs = data['data']
        imgs = data['img']
        if imgs.dim() == 5:
            # b: batch size
            # s: sqeuence length
            # c: channel depth
            # h: height
            # w: width
            b, s, c, h, w = imgs.size()
            # assert np.all([nf == s for nf in num_frames])
            imgs = imgs.view(b * s, c, h, w)
            pids = data['pid']
            pids = pids.view(b, 1).expand(b, s)
            pids = pids.contiguous().view(b * s)
        return imgs, graphs, pids

    def parse_data_for_eval(self, data):
        imgs = data['img']
        graphs = data['data']
        pids = data['pid']
        camids = data['camid']
        return (imgs, graphs), pids, camids

    def extract_features(self, inputs):
        # b: batch size
        # s: sqeuence length
        # c: channel depth
        # h: height
        # w: width
        imgs, graphs = inputs
        b, s, c, h, w = imgs.size()
        imgs = imgs.view(b * s, c, h, w)
        features = self.model(imgs, graphs)
        feature_dim = features.shape[-1]
        features = features.view(b, -1, feature_dim)
        if self.pooling_method == 'avg':
            features = torch.mean(features, 1)
        else:
            features = torch.max(features, 1)[0]
        return features

    def forward_backward(self, data):
        imgs, graphs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda(non_blocking=True)
            graphs = graphs.cuda(non_blocking=True)
            pids = pids.cuda(non_blocking=True)

        # with autocast():
        #     outputs = self.model(imgs, graphs)
        #     loss = self.compute_loss(self.criterion, outputs, pids)

        # self.optimizer.zero_grad()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        
        outputs = self.model(imgs, graphs)
        loss = self.compute_loss(self.criterion, outputs, pids)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_summary = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_summary


class PoseWVideoSoftmaxMSEEngine(PoseWVideoSoftmaxEngine):

    def forward_backward(self, data):
        imgs, graphs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda(non_blocking=True)
            graphs = graphs.cuda(non_blocking=True)
            pids = pids.cuda(non_blocking=True)

        # with autocast():
        #     outputs = self.model(imgs, graphs)
        #     loss = self.compute_loss(self.criterion, outputs, pids)

        # self.optimizer.zero_grad()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        
        outputs, mse = self.model(imgs, graphs)
        loss_softmax = self.compute_loss(self.criterion, outputs, pids)
        
        loss = loss_softmax + mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_summary = {
            'loss': loss.item(),
            'loss_softmax': loss_softmax.item(),
            'loss_mse': mse.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_summary
        