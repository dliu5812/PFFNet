

import cv2
import numpy as np
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import tifffile as tiff

from inference.cell_predictor import CellDemo

from maskrcnn_benchmark.modeling.detector import build_detection_model

import os

from inference.metrics import mask2out, removeoverlap


def select_test_folder(split):

    # Images and annotations for each split on the 3-fold cross-validation (s1, s2, s3).
    # E.g., if you are training with s1 and s2, your validation/testing set should be s3.
    val_root_s1s2 = ''
    gt_root_s1s2 = ''

    val_root_s2s3 = ''
    gt_root_s2s3 = ''

    val_root_s3s1 = ''
    gt_root_s3s1 = ''

    if split == 's1s2':
        val_root_name, gt_folder = val_root_s1s2, gt_root_s1s2
    elif split == 's2s3':
        val_root_name, gt_folder = val_root_s2s3, gt_root_s2s3
    elif split == 's3s1':
        val_root_name, gt_folder = val_root_s3s1, gt_root_s3s1
    else:
        val_root_name, gt_folder = '', ''

    return val_root_name, gt_folder


#
from maskrcnn_benchmark.config import cfg

def infer_cvppp(wts_root, out_pred_root):

    config_file = "../configs/biomed_seg/e2e_mask_rcnn_R_101_FPN_1x_gn-cvppp.yaml"


    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    model = build_detection_model(cfg)
    mkdir(out_pred_root)

    # For the 3-fold validation on TNBC dataset, it is important to add the split name like 's1s2' in the root of the trained weights, if you want to use this code.
    if 's1s2' in wts_root:
        split = 's1s2'
    elif 's2s3' in wts_root:
        split = 's2s3'
    else:
        split = 's3s1'

    print(split)

    cell_demo = CellDemo(
        cfg,
        min_image_size=512,
        confidence_threshold=0.6,
        weight=wts_root,
        model=model
    )

    test_root_name, gt_folder = select_test_folder(split)
    test_imgs = os.listdir(test_root_name)

    for img_name in test_imgs:
        if img_name.endswith(".png"):
            image = cv2.imread(os.path.join(test_root_name, img_name))

            # compute predictions
            predictions, mask_list = cell_demo.run_on_opencv_image(image)
            masks_no_overlap, bi_map, num_mask = removeoverlap(mask_list)

            out_name = os.path.join(out_pred_root, img_name.split('.')[0] + '.tif')

            pred_ins = mask2out(masks_no_overlap, num_mask)

            tiff.imsave(out_name, pred_ins)

            cv2.imwrite(os.path.join(out_pred_root, img_name), predictions)
            cv2.imwrite(os.path.join(out_pred_root, 'bi_mask_' + img_name), (bi_map*255).astype(np.uint8))
            tiff.imsave(out_name, pred_ins)


if __name__ == "__main__":
    wts_root = ''
    out_pred_root = ''

    infer_cvppp(wts_root, out_pred_root)