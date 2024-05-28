import os
import numpy as np
from PIL import Image
from predict_adapter import rgb2ycbcr, calculate_psnr


def evaluate_with_lrhr_pair(gt_path, lr_path, scale):
    # 载入测试图片，以及高清原图
    gt_img_path = [os.path.join(gt_path, x) for x in os.listdir(gt_path) if x.endswith('.png')]
    lr_img_path = [os.path.join(lr_path, x) for x in os.listdir(lr_path) if x.endswith('.png')]

    gt_img_path = sorted(gt_img_path)
    lr_img_path = sorted(lr_img_path)

    avg_psnr = []
    for i in range(len(gt_img_path)):
        gt = Image.open(gt_img_path[i]).convert('RGB')
        lr = Image.open(lr_img_path[i]).convert('RGB')

        gt = np.array(gt)
        lr = np.array(lr)

        gt = gt.astype(np.float32) / 255.
        lr = lr.astype(np.float32) / 255.

        try:
            assert gt.shape == lr.shape
        except AssertionError:
            print('Shape mismatch: gt: {}, lr: {}'.format(gt.shape, lr.shape))
            # Force the shape to be the same
            gt = gt[:lr.shape[0], :lr.shape[1]]

        lr = rgb2ycbcr(lr, only_y=True)
        gt = rgb2ycbcr(gt, only_y=True)

        psnr = calculate_psnr(lr*255, gt*255, border=scale)

        avg_psnr.append(psnr)
        print('PSNR: {:.2f}'.format(psnr))

    print('Avg PSNR: {:.2f}'.format(sum(avg_psnr) / len(avg_psnr)))

gt_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019_hr'
lr_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019swinirx4'
scale = 2
evaluate_with_lrhr_pair(gt_path, lr_path, scale)


# swinir 24.14
