

import os
import numpy as np
import tifffile as tiff
from inference.metrics import remap_label, get_fast_pq, BestDice, FGBGDice
import xlwt


def evaluate_cvppp(pred_root, gt_root):

    sbd_list = []
    dice_list = []
    pq_list = []

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Test Sheet')

    counter = 0

    ws.write(0, 0, 'img_name')
    ws.write(0, 1, 'sbd')
    ws.write(0, 2, 'dice')
    ws.write(0, 3, 'pq')


    test_imgs = os.listdir(pred_root)


    #for img_name in img_names:

    for img_name in test_imgs:
        if img_name.endswith(".tif"):
            gt_name = img_name
            pred_ins = tiff.imread(os.path.join(pred_root, img_name))
            gt_ins = tiff.imread(os.path.join(gt_root, gt_name))

            sbd_score = BestDice(pred_ins, gt_ins)
            dice_score = FGBGDice(pred_ins, gt_ins)

            gt = remap_label(gt_ins, by_size=False)
            pred = remap_label(pred_ins, by_size=False)

            pq_info_cur = get_fast_pq(gt, pred, match_iou=0.5)[0]
            pq_cur = pq_info_cur[2]

            sbd_list.append(sbd_score)
            dice_list.append(dice_score)
            pq_list.append(pq_cur)

            counter = counter + 1

            ws.write(counter, 0, img_name)
            ws.write(counter, 1, sbd_score)
            ws.write(counter, 2, dice_score)
            ws.write(counter, 3, pq_cur)

    wb.save(pred_root + '.xls')

    sbd_array = np.asarray(sbd_list, dtype=np.float32)
    dice_array = np.asarray(dice_list, dtype=np.float32)
    pq_array = np.asarray(pq_list, dtype=np.float32)

    sbd_avg = np.average(sbd_array)
    sbd_std = np.std(sbd_array)

    dice_avg = np.average(dice_array)
    dice_std = np.std(dice_array)

    pq_avg = np.average(pq_array)
    pq_std = np.std(pq_array)

    print(pred_root)

    print('average sbd score of this method is: ', sbd_avg, ' ', sbd_std)
    print('average dice score of this method is: ', dice_avg, ' ', dice_std)
    print('average pq score of this method is: ', pq_avg, ' ',pq_std)


if __name__ == "__main__":
    pred_root = ''
    gt_root = ''

    evaluate_cvppp(pred_root, gt_root)
