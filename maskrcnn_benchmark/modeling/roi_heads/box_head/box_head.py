# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .inference import make_roi_box_ff_processor


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.ff_box_processor = make_roi_box_ff_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.box_logit_resolution = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
        self.fc_bbranch = make_fc(cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.box_logit_resolution ** 2)  # , use_gn=cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.cfg = cfg

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x_pred = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x_pred)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x_pred, result, {}, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        bbox_tr = self.ff_box_processor((class_logits, box_regression), proposals)

        bbox_tr = bbox_tr[0].convert("xyxy")


        return (
            x_pred,
            proposals,
            [bbox_tr],
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
