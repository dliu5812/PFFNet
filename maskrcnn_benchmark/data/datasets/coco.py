# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.config import cfg

import os
import cv2
from PIL import Image


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def gt_name_to_label(img_name):

    sem_path = cfg.MODEL.PANOPTIC.SEM_PATH
    sem_path_cnt = cfg.MODEL.PANOPTIC.SEM_PATH + '_contour2'
    # sem_gt = cv2.imread(os.path.join(sem_path, img_name))[:,:,0]
    sem_gt = Image.open(os.path.join(sem_path, img_name))
    try:
        os.stat(os.path.dirname(sem_path_cnt + '/'))
    except:
        return sem_gt, sem_gt
    sem_gt_cnt = Image.open(os.path.join(sem_path_cnt, img_name))
    # label_mask = (sem_gt > 0).astype(int)
    # lbl_tensor = torch.from_numpy(label_mask).long()
    # sem_gt = Variable(lbl_tensor.unsqueeze(0).cuda(0))
    return sem_gt, sem_gt_cnt


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        img_id = self.id_to_img_map[idx]
        img_name = self.coco.imgs[img_id]['file_name']

        semgt, semgt_cnt = gt_name_to_label(img_name)


        if self._transforms is not None:
            img, target, semgt, semgt_cnt = self._transforms(img, target, semgt, semgt_cnt)

        # print('img', img)
        # print('semgt', semgt)
        # semgt, sempath = gt_name_to_label(img_name)
        # print('semgt size', semgt.size)
        # h, w = img.size()[1], img.size()[2]
        # sem_gt_list = torch.zeros(1, h, w).long()
        # sem_gt_list[0] = bisem_tensor
        # semgt_new = torch.cat([semgt, semgt_cnt], dim=0)
        # print(semgt_new.size())

        return img, target, semgt

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
