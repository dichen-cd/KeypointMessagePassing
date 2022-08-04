import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .siren import SineLayer


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class AttGCNConv(GCNConv):

    def forward(self, x, edge_index, att, *args, **kwargs):
        return super(AttGCNConv, self).forward(x, edge_index, *args, **kwargs) * att


class JRPP(nn.Module):
    '''
    Joints Relationship Pyramid Pooling (JRPP)
    '''
    num_scales = {
        'G1': 1, 'G2': 2, 'G3': 3,
        'G4': 5, 'G5': 12, 'G6': 17
    }

    num_scales_headless = {
        'G1': 1, 'G2': 2, 'G3': 2, 'G7': 2
    }

    def __init__(self, modes, num_keypoints=17):
        super(JRPP, self).__init__()
        assert set(modes).issubset({'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'})
        self.modes = modes
        self.num_keypoints = num_keypoints

    @property
    def output_scales(self):
        if self.num_keypoints == 17:
            return sum([self.num_scales[m] for m in self.modes])
        elif self.num_keypoints == 12:
            return sum([self.num_scales_headless[m] for m in self.modes])

    def forward(self, x):
        '''
        x: torch.Tensor, N * d with d being the feature dim (256)
           N % 17 =0
        '''
        N, d = x.shape
        x = x.view(-1, self.num_keypoints, d)
        feats = []
        for m in self.modes:
            feats.append(self.apply_pool(x, m))
        return torch.cat(feats, dim=1)

    def apply_pool(self, x, mode):
        '''
        x: torch.Tensor, N * 17 * d
        '''
        if mode == 'G1':
            return self.G1(x)
        elif mode == 'G2':
            return self.G2(x)
        elif mode == 'G3':
            return self.G3(x)
        elif mode == 'G7':
            return self.G7(x)
        else:
            raise NotImplementedError

    def G1(self, x):
        '''
        x: torch.Tensor, N * 17 * d
        '''
        return x.mean(dim=1, keepdim=True)

    def G2(self, x):
        if self.num_keypoints == 17:
            upper = x[:, :11, :].mean(dim=1, keepdim=True)
            lower = x[:, 11:, :].mean(dim=1, keepdim=True)
            return torch.cat([upper, lower], dim=1)
        elif self.num_keypoints == 12:
            upper = x[:, :6, :].mean(dim=1, keepdim=True)
            lower = x[:, 6:, :].mean(dim=1, keepdim=True)
            return torch.cat([upper, lower], dim=1)

    def G3(self, x):
        if self.num_keypoints == 17:
            head = x[:, [0, 1, 2, 3, 4], :].mean(dim=1, keepdim=True)
            leftarm_rightleg = \
                x[:, [5, 7, 9, 12, 14, 16], :].mean(dim=1, keepdim=True)
            rightarm_leftleg = \
                x[:, [6, 8, 10, 11, 13, 15], :].mean(dim=1, keepdim=True)
            return torch.cat([head, leftarm_rightleg, rightarm_leftleg], dim=1)
        elif self.num_keypoints == 12:
            # no head
            leftarm_rightleg = \
                x[:, [0, 2, 4, 7, 9, 11], :].mean(dim=1, keepdim=True)
            rightarm_leftleg = \
                x[:, [1, 3, 5, 6, 8, 10], :].mean(dim=1, keepdim=True)
            return torch.cat([leftarm_rightleg, rightarm_leftleg], dim=1)

    def G7(self, x):
        if self.num_keypoints == 17:
            raise NotImplementedError
        elif self.num_keypoints == 12:
            leftarm_leftleg = \
                x[:, [0, 2, 4, 6, 8, 10], :].mean(dim=1, keepdim=True)
            rightarm_rightleg = \
                x[:, [1, 3, 5, 7, 9, 11], :].mean(dim=1, keepdim=True)
            return torch.cat([leftarm_leftleg, rightarm_rightleg], dim=1)

    # G4, G5, G6 not necessarily better so let's leave them for now.


class JointsGait(nn.Module):

    def __init__(self, num_classes,
                 loss='softmax',
                 pretrained=False,
                 use_gpu=False,
                 feature_dim=256,
                 pool_modes=['G1', 'G2', 'G3']):
        super(JointsGait, self).__init__()

        self.feature_dim = feature_dim

        omega = 30.0
        self.conv1 = AttGCNConv(2, 64)
        self.ac1 = nn.ReLU(inplace=True)
        # self.ac1 = SineLayer(64, 64, is_first=False, omega_0=omega)
        self.conv2 = AttGCNConv(64, 64)
        self.ac2 = nn.ReLU(inplace=True)
        # self.ac2 = SineLayer(64, 64, is_first=False, omega_0=omega)
        self.conv3 = AttGCNConv(64, 64)
        self.ac3 = nn.ReLU(inplace=True)
        # self.ac3 = SineLayer(64, 64, is_first=False, omega_0=omega)

        self.conv4 = AttGCNConv(64, 128)
        self.ac4 = nn.ReLU(inplace=True)
        # self.ac4 = SineLayer(128, 128, is_first=False, omega_0=omega)
        self.conv5 = AttGCNConv(128, 128)
        self.ac5 = nn.ReLU(inplace=True)
        # self.ac5 = SineLayer(128, 128, is_first=False, omega_0=omega)
        self.conv6 = AttGCNConv(128, 128)
        self.ac6 = nn.ReLU(inplace=True)
        # self.ac6 = SineLayer(128, 128, is_first=False, omega_0=omega)

        self.conv7 = AttGCNConv(128, 256)
        self.ac7 = nn.ReLU(inplace=True)
        # self.ac7 = SineLayer(256, 256, is_first=False, omega_0=omega)
        self.conv8 = AttGCNConv(256, 256)
        self.ac8 = nn.ReLU(inplace=True)
        # self.ac8 = SineLayer(256, 256, is_first=False, omega_0=omega)
        self.conv9 = AttGCNConv(256, 256)
        self.ac9 = nn.ReLU(inplace=True)
        # self.ac9 = SineLayer(256, 256, is_first=False, omega_0=omega)

        self.pool = JRPP(modes=pool_modes)
        self.pool_scales = self.pool.output_scales

        self.feat_fcs = nn.ModuleList(
            [nn.Linear(256, self.feature_dim) for _ in range(self.pool_scales)])
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
        x = inputs[:, :2]  # size = (N, 2)
        att = inputs[:, 2:]  # size = (N, 1)

        x = self.gcn_features(x, edge_index, att)
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

    def gcn_features(self, x, edge_index, att):
        x = self.conv1(x, edge_index, att)
        x = self.ac1(x)
        x = self.conv2(x, edge_index, att)
        x = self.ac2(x)
        x = self.conv3(x, edge_index, att)
        x = self.ac3(x)

        x = self.conv4(x, edge_index, att)
        x = self.ac4(x)
        x = self.conv5(x, edge_index, att)
        x = self.ac5(x)
        x = self.conv6(x, edge_index, att)
        x = self.ac6(x)

        x = self.conv7(x, edge_index, att)
        x = self.ac7(x)
        x = self.conv8(x, edge_index, att)
        x = self.ac8(x)
        x = self.conv9(x, edge_index, att)
        x = self.ac9(x)
        # x.size = (N, 256)

        # size = (N/17, k (6), 256), k depends on the pooling mode
        x = self.pool(x)
        return x
