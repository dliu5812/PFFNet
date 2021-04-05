import torch
from maskrcnn_benchmark.layers.misc import interpolate
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform


class ROIFeatureFusion(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIFeatureFusion, self).__init__()
        self.cfg = cfg
        num_class = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.sem_block_1x1 = conv_with_kaiming_uniform()(num_class, num_class, 1, 1)
        self.sem_block_3x3 = conv_with_kaiming_uniform()(num_class, num_class, 3, 1)


    def forward(self, masks, boxes, feature_map, padding=1):



        assert self.cfg.MODEL.PANOPTIC.PS_ON == True

        for mask, box in zip(masks.unsqueeze(1), boxes.bbox):
            box = box.numpy().astype(np.int32)
            _, _, im_h, im_w = feature_map.size()
            TO_REMOVE = 1
            w = box[2] - box[0] + TO_REMOVE
            h = box[3] - box[1] + TO_REMOVE
            w = max(w, 1)
            h = max(h, 1)
            resized_mask = interpolate(mask, [w, h], mode='bilinear')

            x_0 = box[0]
            x_1 = min(box[2] + 1, im_w)
            y_0 = box[1]
            y_1 = min(box[3] + 1, im_h)

            mask_cropped = resized_mask[:, :,0: min(w, im_w - y_0), 0: min(h, im_h - x_0)]


            if not self.cfg.MODEL.PANOPTIC.ATTENTION_FUSION:
                feature_map[:, :, y_0: min(y_0 + w, im_w), x_0: min(x_0 + h, im_h)] = mask_cropped
            else:
                mask_cropped_att = mask_cropped.sigmoid().detach()
                feature_map[:, :, y_0: min(y_0 + w, im_w), x_0: min(x_0 + h, im_h)] *= (1 + mask_cropped_att)

        if not self.cfg.MODEL.PANOPTIC.ATTENTION_FUSION:
            out_sem_feature_fusion = feature_map
        else:
            out_sem_feature_fusion = self.sem_block_3x3(F.relu(feature_map))
            out_sem_feature_fusion = self.sem_block_1x1(F.relu(out_sem_feature_fusion))

        return out_sem_feature_fusion


def build_roi_feature_fusion(cfg):
    return ROIFeatureFusion(cfg)