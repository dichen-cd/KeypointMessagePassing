import torch
from torch import nn
from torch.nn import LayerNorm, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GENConv, DeepGCNLayer

from .joints_gait import weights_init_kaiming, weights_init_classifier, JRPP


class DeeperGCN(nn.Module):

    def __init__(self, num_classes,
                 loss='softmax',
                 pretrained=False,
                 use_gpu=False,
                 # model-specific kwargs:
                 input_dim=2,
                 hidden_channels=64,
                 num_layers=56,
                 feature_dim=256,
                 pool_modes=['G1', 'G2', 'G3'],
                 num_keypoints=17):
        super(DeeperGCN, self).__init__()

        self.feature_dim = feature_dim

        self.node_encoder = nn.Linear(input_dim, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=False)
            self.layers.append(layer)

        self.pool = JRPP(modes=pool_modes, num_keypoints=num_keypoints)
        self.pool_scales = self.pool.output_scales

        self.feat_fcs = nn.ModuleList(
            [nn.Linear(hidden_channels, self.feature_dim) for _ in range(self.pool_scales)])
        for m in self.feat_fcs:
            m.apply(weights_init_kaiming)

        self.feat_bns = nn.ModuleList(
            [nn.BatchNorm1d(self.feature_dim) for _ in range(self.pool_scales)])
        for m in self.feat_bns:
            m.bias.requires_grad_(False)  # no shift
            m.apply(weights_init_kaiming)

        self.classifier = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_classes, bias=False)
             for _ in range(self.pool_scales)])
        for m in self.classifier:
            m.apply(weights_init_classifier)

    def forward(self, data):
        inputs, edge_index = data.x, data.edge_index
        x = inputs[:, :-1]  # size = (N, dim)
        # att = inputs[:, -1:]  # size = (N, 1)

        x = self.gcn_features(x, edge_index)
        N_p, p_s, _ = x.shape

        f = []
        for i in range(self.pool_scales):
            ft = self.feat_fcs[i](x[:, i, :])  # for triplet loss
            fi = self.feat_bns[i](ft)  # for softmax
            f.append(fi)

        if not self.training:
            output_f = torch.cat(f, dim=1)
            return F.normalize(output_f, p=2, dim=1)

        y = []
        for i in range(self.pool_scales):
            y_i = self.classifier[i](f[i])
            y.append(y_i)

        return y

    def gcn_features(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.pool(x)
        return x
