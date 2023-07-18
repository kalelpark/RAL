import torch.nn as nn
import torch.nn.functional as F 
from .anchors import generate_default_anchor_maps
from .anchors import *

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class customModel(nn.Module):
    def __init__(self, backbone, feature_size, classes_num):
        super(customModel, self).__init__()
        self.backbone = backbone
        self.num_ftrs = 1024
        self.proposal_net = ProposalNet(depth = self.num_ftrs)
        self.pad_side = 512
        self.topn = 1
    
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.edge_anchors = (edge_anchors+self.pad_side).astype(np.int64)

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        # stage 2
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        # stage 3
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
        
        # concat features from different stages
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2 * 3),
            nn.Linear(self.num_ftrs//2 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )
    
    def forward(self, x):
        _, f1, f2, f3 = self.backbone(x)
        # torch.Size([1, 256, 128, 128]) torch.Size([1, 512, 64, 64]) torch.Size([1, 1024, 32, 32])
        batch = x.shape[0]
        rpn_score = self.proposal_net(f3.detach())
        all_cdds = [np.concatenate((x.reshape(-1, 1), 
                    self.edge_anchors.copy(),
                    np.arange(0, len(x)).reshape(-1, 1)), 
                    axis=1) for x in rpn_score.data.cpu().numpy()]

        top_n_cdds = np.array([hard_nms(x, self.topn, iou_thresh=0.4) for x in all_cdds])
        top_n_index = top_n_cdds[:, :, -1].astype(np.int64)
        top_n_index = torch.from_numpy(top_n_index).long().to(x.device)
        
        # re-input salient parts
        part_imgs = torch.zeros([batch, self.topn, 3, 512, 512]).to(x.device)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        for i in range(batch):
            for j in range(self.topn):
                [y0, x0, y1, x1] = top_n_cdds[i, j, 1:5].astype(np.int64)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], 
                                                        size=(512, 512), mode='bilinear',
                                                        align_corners=True)

        part_imgs = part_imgs.view(batch*self.topn, 3, 512, 512)
        _, f1_part, f2_part, f3_part = self.backbone(part_imgs.detach())

        # torch.Size([1, 256, 128, 128]) torch.Size([1, 512, 64, 64]) torch.Size([1, 1024, 32, 32])
        # torch.Size([4, 128]) torch.Size([4, 128]) torch.Size([4, 128])        
        # print("f1 : ", f1_part.size(), f2_part.size(), f3_part.size())

        f1_part = self.conv_block1(f1_part).view(batch*self.topn, -1)
        f2_part = self.conv_block2(f2_part).view(batch*self.topn, -1)
        f3_part = self.conv_block3(f3_part).view(batch*self.topn, -1)
        

        yp1 = self.classifier1(f1_part)
        yp2 = self.classifier2(f2_part)
        yp3 = self.classifier3(f3_part)
        yp4 = self.classifier_concat(torch.cat((f1_part, f2_part, f3_part), -1))

        f1 = self.conv_block1(f1).view(batch, -1)
        f2 = self.conv_block2(f2).view(batch, -1)
        f3 = self.conv_block3(f3).view(batch, -1)
        y1 = self.classifier1(f1)
        y2 = self.classifier2(f2)
        y3 = self.classifier3(f3)
        y4 = self.classifier_concat(torch.cat((f1, f2, f3), -1))

        return y1, y2, y3, y4, yp1, yp2, yp3, yp4