# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..roi_heads.feature_fusion import build_roi_feature_fusion



class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.cfg = cfg
        self.sem_consistence = cfg.MODEL.PANOPTIC.SEMANTIC_CONSISTENCE
        self.sem_consistence_lambda = cfg.MODEL.PANOPTIC.SEMANTIC_CONSISTENCE_LAMBDA


        if not cfg.MODEL.PANOPTIC.PS_ON:
            self.backbone = build_backbone(cfg)
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        else:
            self.backbone, self.semseg = build_backbone(cfg)
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
            self.roi_feature_fusion = build_roi_feature_fusion(cfg)


    def forward(self, images, targets=None, semgt = None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features_all = self.backbone(images.tensors)

        # print('images shape', images.tensors.size())

        if not self.cfg.MODEL.PANOPTIC.PS_ON:
            features = features_all
        else:
            features = features_all[1:]
            fpn_semantic_pred = features_all[0]
            pred_sembranch = self.semseg(images.tensors)

        proposals, proposal_losses, feat_rpn_scales = self.rpn(images, features, targets)
        if self.roi_heads:
            if not self.cfg.MODEL.PANOPTIC.PS_ON:
                x, result, detector_losses = self.roi_heads(features, proposals, targets)
            else:
                x, result, detector_losses, masks, bboxes = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            if not self.cfg.MODEL.PANOPTIC.PS_ON:
                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)
                return losses


            gt_semseg = semgt

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            # sem branch

            sembranch_loss = nn.CrossEntropyLoss()(pred_sembranch, gt_semseg)
            sembranch_losses = {'loss_sem_branch': sembranch_loss}
            losses.update(sembranch_losses)

            #  mask feature fusion

            pred_maskfusion = self.roi_feature_fusion(masks, bboxes, fpn_semantic_pred)

            maskfusion_loss = nn.CrossEntropyLoss()(pred_maskfusion, gt_semseg)

            maskfusion_losses = {'loss_mask_fusion': maskfusion_loss}
            losses.update(maskfusion_losses)


            if not self.sem_consistence:
                return losses

            # semantic feature consistency

            prob_sembranch = nn.Softmax(dim=1)(pred_sembranch)
            prob_maskfusion = nn.Softmax(dim=1)(pred_maskfusion)

            sem_consistence_loss = nn.MSELoss()(prob_sembranch, prob_maskfusion) * self.sem_consistence_lambda
            sem_consistence_losses = {'loss_sem_consistence': sem_consistence_loss}
            losses.update(sem_consistence_losses)
            return losses

        return result
