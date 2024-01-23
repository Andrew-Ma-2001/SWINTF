# 1 Load in model
# 2 Predict the input image list
# 3 Calculate the PSNR

import os
from PIL import Image
import torch
import numpy as np
from utils.utils_image import permute_squeeze, calculate_psnr, imresize_np
from nets.swinir import SwinIR
import yaml


def evaluate_with_lrhr_pair(gt_path, lr_path, model):
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

        gt = torch.from_numpy(np.ascontiguousarray(np.transpose(gt, (2, 0, 1)))).float()
        lr = torch.from_numpy(np.ascontiguousarray(np.transpose(lr, (2, 0, 1)))).float()

        gt = gt.unsqueeze(0)
        lr = lr.unsqueeze(0)

        # print(gt.shape)
        # print(lr.shape)

        # 预测
        with torch.no_grad():
            sr = model(lr)

        # 计算PSNR
        sr = permute_squeeze(sr)
        gt = permute_squeeze(gt)

        psnr = calculate_psnr(sr * 255, gt * 255, border=scale)
        avg_psnr.append(psnr)
        print('PSNR: {:.2f}'.format(psnr))

    print('Avg PSNR: {:.2f}'.format(sum(avg_psnr) / len(avg_psnr)))

def evaluate_with_hr(gt_path, model):
    gt_img_path = [os.path.join(gt_path, x) for x in os.listdir(gt_path) if x.endswith('.png')]
 
    gt_img_path = sorted(gt_img_path)

    avg_psnr = []
    for i in range(len(gt_img_path)):
        gt = Image.open(gt_img_path[i]).convert('RGB')

        gt = np.array(gt)

        gt = gt.astype(np.float32) / 255.
        lr = imresize_np(gt, 1/scale, True)

        gt = torch.from_numpy(np.ascontiguousarray(np.transpose(gt, (2, 0, 1)))).float()
        lr = torch.from_numpy(np.ascontiguousarray(np.transpose(lr, (2, 0, 1)))).float()

        gt = gt.unsqueeze(0)
        lr = lr.unsqueeze(0)

        # print(gt.shape)
        # print(lr.shape)

        # 预测
        with torch.no_grad():
            sr = model(lr)

        # 计算PSNR
        sr = permute_squeeze(sr)
        gt = permute_squeeze(gt)

        psnr = calculate_psnr(sr * 255, gt * 255, border=scale)
        avg_psnr.append(psnr)
        print('PSNR: {:.2f}'.format(psnr))
    print('Avg PSNR: {:.2f}'.format(sum(avg_psnr) / len(avg_psnr)))



if __name__ == '__main__':
    config_path = 'config/example.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    gpu_ids = config['train']['gpu_ids']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    # 载入模型
    model_path = 'experiments/SwinIR_20231221_161712/95000_model.pth'

    scale = config['train']['scale']
    model = SwinIR(upscale=config['network']['upsacle'], 
                    in_chans=config['network']['in_channels'],
                    img_size=config['network']['image_size'],
                    window_size=config['network']['window_size'],
                    img_range=config['network']['image_range'],
                    depths=config['network']['depths'],
                    embed_dim=config['network']['embed_dim'],
                    num_heads=config['network']['num_heads'],
                    mlp_ratio=config['network']['mlp_ratio'],
                    upsampler=config['network']['upsampler'],
                    resi_connection=config['network']['resi_connection'])


    model.eval()
    model.cuda()
    model = torch.nn.DataParallel(model) 
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print('Resume from checkpoint from {}'.format(model_path))


    # gt_path = 'dataset/testsets/Set14/GTmod12'
    # lr_path = 'dataset/testsets/Set14/LRbicx4'
    # # scale = 4
    # evaluate_with_lrhr_pair(gt_path, lr_path, model)

    gt_path = 'dataset/testsets/urban100'
    evaluate_with_hr(gt_path, model)