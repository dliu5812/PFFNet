# Mask Scoring R-CNN
# Wriiten by zhaojin.huang, 2018-12.

import torch
from torch import nn
from torch.nn import functional as F


from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from maskrcnn_benchmark.modeling.make_layers import make_fc


class MaskIoUFeatureExtractor(nn.Module):
    """
    MaskIou head feature extractor.
    """

    def __init__(self, cfg):
        super(MaskIoUFeatureExtractor, self).__init__()
        
        input_channels = 260  # default 257
        use_gn = cfg.MODEL.MASKIOU_USE_GN


        self.maskiou_fcn1 = make_conv3x3(input_channels, 256, use_gn= use_gn)
        self.maskiou_fcn2 = make_conv3x3(256, 256, use_gn= use_gn)
        self.maskiou_fcn3 = make_conv3x3(256, 256, use_gn= use_gn)
        self.maskiou_fcn4 = make_conv3x3(256, 256, stride=2, use_gn= use_gn)
        self.maskiou_fc1 = make_fc(256*7*7, 1024, use_gn= use_gn)
        self.maskiou_fc2 = make_fc(1024, 1024, use_gn= use_gn)



    def forward(self, x, mask):


        roi_num,_, roi_h, roi_w = x.shape
        mask_rs = mask.view(roi_num, -1, roi_h, roi_w)

        x = torch.cat((x, mask_rs), 1)
        x = F.relu(self.maskiou_fcn1(x))
        x = F.relu(self.maskiou_fcn2(x))
        x = F.relu(self.maskiou_fcn3(x))
        x = F.relu(self.maskiou_fcn4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
 
        return x


def make_roi_maskiou_feature_extractor(cfg):
    func = MaskIoUFeatureExtractor
    return func(cfg)
