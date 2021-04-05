# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling import registry


@registry.ROI_MASK_PREDICTOR.register("MaskRCNNC4Predictor")
class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels
        self.dual_modal = cfg.MODEL.ROI_MASK_HEAD.DUAL_MODAL
        self.use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN

        if self.dual_modal:
            self.conv5_mask = Conv2d(num_inputs, dim_reduced * 4, 3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        else:
            self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)

        self.mask_fcn_final = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        self.cfg = cfg

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


    def forward(self, feature_fcn_conv4, mask_fc_feature):
        feature_fcn_conv4 = F.relu(self.conv5_mask(feature_fcn_conv4))
        if not self.dual_modal:
            return self.mask_fcn_final(feature_fcn_conv4)

        mask_fcn_logit = self.pixel_shuffle(feature_fcn_conv4)
        mask_fcn_out = self.mask_fcn_final(mask_fcn_logit)
        mask_fc_resize = mask_fc_feature.view(-1, 1, self.cfg.MODEL.ROI_MASK_HEAD.RESOLUTION, self.cfg.MODEL.ROI_MASK_HEAD.RESOLUTION)
        mask_fc_out = mask_fc_resize.repeat(1, self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, 1, 1)
        out = mask_fc_out + mask_fcn_out
        return out



@registry.ROI_MASK_PREDICTOR.register("MaskRCNNConv1x1Predictor")
class MaskRCNNConv1x1Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNConv1x1Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_inputs = in_channels

        self.mask_fcn_logits = Conv2d(num_inputs, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.mask_fcn_logits(x)


def make_roi_mask_predictor(cfg, in_channels):
    func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg, in_channels)
