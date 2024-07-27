import os
import numpy as np
from PIL import Image
from predict_adapter import rgb2ycbcr, calculate_psnr
from matplotlib import pyplot as plt

def save_lrhr_pair(gt, lr, save_path='.'):
    # 将lr和gt保存到save_path
    plt.imsave(os.path.join(save_path, 'lr.png'), lr)
    plt.imsave(os.path.join(save_path, 'gt.png'), gt)


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

        # save_lrhr_pair(gt, lr)

        lr = rgb2ycbcr(lr, only_y=True)
        gt = rgb2ycbcr(gt, only_y=True)

        psnr = calculate_psnr(lr*255, gt*255, border=scale)

        avg_psnr.append(psnr)
        print('PSNR: {:.2f}'.format(psnr))

    print('Avg PSNR: {:.2f}'.format(sum(avg_psnr) / len(avg_psnr)))

def evaluate_with_overlap_lrhr_pair(gt_path, lr_path, scale, overlap):
    # 载入测试图片，以及高清原图
    gt_img_path = [os.path.join(gt_path, x) for x in os.listdir(gt_path) if x.endswith('.png')]
    lr_img_path = [os.path.join(lr_path, x) for x in os.listdir(lr_path) if x.endswith('.png')]

    gt_img_path = sorted(gt_img_path)[:10]
    lr_img_path = sorted(lr_img_path)[:10]

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
            gt = gt[overlap:gt.shape[0]-overlap, overlap:gt.shape[1]-overlap]
            # Force the shape to be the same
        gt = gt[:lr.shape[0], :lr.shape[1]]
        save_lrhr_pair(gt, lr)

        lr = rgb2ycbcr(lr, only_y=True)
        gt = rgb2ycbcr(gt, only_y=True)

        psnr = calculate_psnr(lr*255, gt*255, border=scale)

        avg_psnr.append(psnr)
        print('PSNR: {:.2f}'.format(psnr))

    print('Avg PSNR: {:.2f}'.format(sum(avg_psnr) / len(avg_psnr)))

gt_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019_hr'
# lr_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019_overlap0_x4'
lr_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019_43w_overlap0_x4'
# lr_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019swinir_x4'
scale = 2
overlap = 8
evaluate_with_lrhr_pair(gt_path, lr_path, scale)
# evaluate_with_overlap_lrhr_pair(gt_path, lr_path, scale, overlap)

# PSNR
# swinir 24.14 (baseline)
# adapter overlap 0 24.15 23wStep
# adapter overlap 0 24.14 43wStep 

# 加速对比下 
# 前10张 24.79 scale*4 border
# 前10张 24.81 scale border
# 前10张 swinir 24.79  scale border
# 前10张 swinir 24.78  scale*4 border