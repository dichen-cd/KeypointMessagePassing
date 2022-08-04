import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchreid.utils import load_pretrained_weights

from torchreid.models import pcb_p4, pcb_p6, resnet50
from .deeper_gcn import DeeperGCN
from .joints_gait import weights_init_kaiming


class IdentityLayer(nn.Module):
    
    def forward(self, x):
        return x
        

class KeypointMessagePassing(nn.Module):

    def __init__(self, num_classes,
                 loss='softmax',
                 pretrained=False,
                 use_gpu=False,
                 # model-specific kwargs:
                 input_dim=6,
                 num_keypoints=12,
                 hidden_channels=64,
                 num_stages=5,
                 pcb_pretrained_path='logs/v0.3.2/Jul03_21-53-45_TITAN-RTX/model/model.pth.tar-45',
                 xv_only=False,
                 xg_only=False,
                 pool_modes=['G1', 'G2', 'G3'],
                 ):
        super(KeypointMessagePassing, self).__init__()
        self.num_keypoints = num_keypoints
        self.xv_only = xv_only
        self.xg_only = xg_only
        self.pool_modes = pool_modes
        if self.xv_only:
            print('=> Testing with visual feature...')
        elif self.xg_only:
            print('=> Testing with gcn feature...')
        else:
            print('=> Testing with both visual and gcn features...')

        self.pcb = pcb_p4(num_classes, loss='softmax', pretrained=False)
        # self.pcb = pcb_p6(num_classes, loss='softmax', pretrained=False)
        load_pretrained_weights(self.pcb, pcb_pretrained_path)
        print(f'=> Loaded pretrained PCB model ^ ^')
        self.spatial_scales = [0.25, 0.25, 0.125, 0.0625, 0.0625]

        self.gcn = DeeperGCN(num_classes,
                             loss='softmax',
                             pretrained=False,
                             use_gpu=True,
                             # model-specific kwargs:
                             input_dim=input_dim,
                             hidden_channels=hidden_channels,
                             num_layers=27,
                             feature_dim=256,
                             pool_modes=self.pool_modes,
                             num_keypoints=self.num_keypoints)
        self.gcn.post_act = nn.ReLU(inplace=True)
        self.gcn.post_norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)

        self.fc_dimup = nn.ModuleList()
        # conv1 -> bn1 -> relu-> maxpool
        self.fc_dimup.append(IdentityLayer())
        self.fc_dimup.append(nn.Linear(hidden_channels, 256))   # layer1
        self.fc_dimup.append(nn.Linear(hidden_channels, 512))   # layer2
        self.fc_dimup.append(nn.Linear(hidden_channels, 1024))  # layer3
        self.fc_dimup.append(nn.Linear(hidden_channels, 2048))  # layer4
        for m in self.fc_dimup:
            m.apply(weights_init_kaiming)

        self.fc_dimdown = nn.ModuleList()
        self.fc_dimdown.append(nn.Linear(64, hidden_channels))
        self.fc_dimdown.append(nn.Linear(256, hidden_channels))
        self.fc_dimdown.append(nn.Linear(512, hidden_channels))
        self.fc_dimdown.append(nn.Linear(1024, hidden_channels))
        self.fc_dimdown.append(nn.Linear(2048, hidden_channels))
        for m in self.fc_dimdown:
            m.apply(weights_init_kaiming)

    def forward(self, imgs, graphs):

        if not self.training:
            return self.pcb(imgs)

        _, _, h, w = imgs.shape
        inputs, edge_index = graphs.x, graphs.edge_index
        edge_attr = None
        orig_coords = inputs[:, :2].view(imgs.shape[0], self.num_keypoints, 2)
        # remove orig_coordinates && keypoint confidence. size = (N, dim)
        # x_g = inputs[:, 2:-1]

        pseudo_boxes = self.coords_to_pseudo_box(orig_coords, w, h)

        # Stage 0
        x_v = self.pcb.conv1(imgs)
        x_v = self.pcb.bn1(x_v)
        x_v = self.pcb.relu(x_v)
        x_v = self.pcb.maxpool(x_v)
        x_v, x_g = self.interact(x_v, 0.0, 0, pseudo_boxes)

        # Stage 1
        x_v = self.pcb.layer1(x_v)
        for layer in self.gcn.layers[0:5]:  # 27
            x_g = layer(x_g, edge_index, edge_attr)
        x_v, x_g = self.interact(x_v, x_g, 1, pseudo_boxes)

        # Stage 2
        x_v = self.pcb.layer2(x_v)
        for layer in self.gcn.layers[5:12]:  # 27
            x_g = layer(x_g, edge_index, edge_attr)
        x_v, x_g = self.interact(x_v, x_g, 2, pseudo_boxes)

        # Stage 3
        x_v = self.pcb.layer3(x_v)
        for layer in self.gcn.layers[12:23]:  # 27
            x_g = layer(x_g, edge_index, edge_attr)
        x_v, x_g = self.interact(x_v, x_g, 3, pseudo_boxes)

        # Stage 4
        x_v = self.pcb.layer4(x_v)
        for layer in self.gcn.layers[23:27]:  # 23
            x_g = layer(x_g, edge_index, edge_attr)
        x_v, x_g = self.interact(x_v, x_g, 4, pseudo_boxes)

        # Post process
        x_v = self.pcb_post_process(x_v)
        x_g = self.gcn_post_process(x_g)

        if not self.training:
            if self.xv_only:
                return F.normalize(x_v, p=2, dim=1)
            elif self.xg_only:
                return F.normalize(x_g, p=2, dim=1)
            else:
                output_f = torch.cat([x_v, x_g], dim=1)
                return F.normalize(output_f, p=2, dim=1)
        else:
            return x_v + x_g  # both of them are lists

    def coords_to_pseudo_box(self, orig_coords, w, h):
        N = orig_coords.shape[0]
        boxes = torch.zeros(N, self.num_keypoints, 4).to(orig_coords)
        boxes[:, :, 0] = orig_coords[:, :, 0] * w
        boxes[:, :, 1] = orig_coords[:, :, 1] * h
        boxes[:, :, 2] = boxes[:, :, 0] + 1
        boxes[:, :, 3] = boxes[:, :, 1] + 1
        # if keypoints are annotated as 0 or -1, then replace the keypoint feature with global feature.
        invalid_indices = (boxes[:, :, 0] <= 0) * \
            (boxes[:, :, 1] <= 0)  # x = 0 or y = 0
        boxes[invalid_indices] = torch.tensor([0., 0., w, h]).to(orig_coords)
        # lists, [size = (numkeypoints, 4)]
        return [b.squeeze(0) for b in boxes.split(1, 0)]

    def interact(self, x_v, x_g, stage_idx, pseudo_boxes):
        N, c, h, w = x_v.shape

        x_g = self.fc_dimup[stage_idx](x_g)  # size = (N*num_keypoints, c)

        pooled_x_v = roi_align(x_v, pseudo_boxes,
                               output_size=1,
                               spatial_scale=self.spatial_scales[stage_idx])
        # size = (N*num_keypoints, c, 1, 1)

        fuse = pooled_x_v.view(N * self.num_keypoints, c) + x_g
        # pooled_x_v = fuse

        x_g = self.fc_dimdown[stage_idx](fuse)

        return x_v, x_g

    def pcb_post_process(self, x_v):
        v_g = self.pcb.parts_avgpool(x_v)

        if not self.pcb.training:
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)

        v_g = self.pcb.dropout(v_g)
        v_h = self.pcb.conv5(v_g)

        y = []
        for i in range(self.pcb.parts):
            v_h_i = v_h[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0), -1)
            y_i = self.pcb.classifier[i](v_h_i)
            y.append(y_i)

        if self.pcb.loss == 'softmax':
            return y
        # elif self.pcb.loss == 'triplet':
        #     v_g = F.normalize(v_g, p=2, dim=1)
        #     return y, v_g.view(v_g.size(0), -1)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))

    def gcn_post_process(self, x_g):
        x_g = self.gcn.post_act(self.gcn.post_norm(x_g))
        x_g = F.dropout(x_g, p=0.1, training=self.gcn.training)

        x_g = self.gcn.pool(x_g)

        N_p, p_s, _ = x_g.shape

        f = []
        for i in range(self.gcn.pool_scales):
            ft = self.gcn.feat_fcs[i](x_g[:, i, :])  # for triplet loss
            fi = self.gcn.feat_bns[i](ft)  # for softmax
            f.append(fi)

        if not self.gcn.training:
            output_f = torch.cat(f, dim=1)
            # return F.normalize(output_f, p=2, dim=1)
            return output_f

        y = []
        for i in range(self.gcn.pool_scales):
            y_i = self.gcn.classifier[i](f[i])
            y.append(y_i)

        return y

