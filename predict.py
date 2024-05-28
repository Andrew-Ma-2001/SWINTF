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
from matplotlib import pyplot as plt

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

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

        # output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # 计算PSNR
        sr = permute_squeeze(sr)
        gt = permute_squeeze(gt)
        
        # Save the sr image
        plt.imsave('sr.png', (sr*255).astype(np.uint8))
        # plt.imsave('lr.png', (lr*255).astype(np.uint8))
        plt.imsave('gt.png', (gt*255).astype(np.uint8))



        sr = rgb2ycbcr(sr, only_y=True)
        gt = rgb2ycbcr(gt, only_y=True)

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
    config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109test.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    gpu_ids = config['train']['gpu_ids']
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,6,7'

    # 载入模型
    # model_path = '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20231221_161712/120000_model.pth'
    # SwinIR+SAM
    model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'

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

    # pretrained_model = torch.load(model_path)
    # param_key_g = 'params'
    # model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
    pretrained_model = torch.load(model_path)
    param_key_g = 'params'
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    model = torch.nn.DataParallel(model) 
    
    print('Resume from checkpoint from {}'.format(model_path))

    # gt_path = 'dataset/testsets/Set5/GTmod12'
    lr_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019swinir'
    save_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019swinirx4'
    # scale = 4

    # gt_path = 'dataset/testsets/urban100'
    # evaluate_with_hr(gt_path, model)

    # gt_img_path = [os.path.join(gt_path, x) for x in os.listdir(gt_path) if x.endswith('.png')]
    lr_img_path = [os.path.join(lr_path, x) for x in os.listdir(lr_path) if x.endswith('.png')]

    # gt_img_path = sorted(gt_img_path)
    lr_img_path = sorted(lr_img_path)

    avg_psnr = []
    for i in range(len(lr_img_path)):
        # gt = Image.open(gt_img_path[i]).convert('RGB')
        lr = Image.open(lr_img_path[i]).convert('RGB')

        # gt = np.array(gt)
        lr = np.array(lr)

        # gt = gt.astype(np.float32) / 255.
        lr = lr.astype(np.float32) / 255.

        # gt = torch.from_numpy(np.ascontiguousarray(np.transpose(gt, (2, 0, 1)))).float()
        lr = torch.from_numpy(np.ascontiguousarray(np.transpose(lr, (2, 0, 1)))).float()

        # gt = gt.unsqueeze(0)
        lr = lr.unsqueeze(0)

        # print(gt.shape)
        # print(lr.shape)

        # 预测
        with torch.no_grad():
            sr = model(lr)

        # output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # 计算PSNR
        # sr = permute_squeeze(sr)
        # gt = permute_squeeze(gt)
        sr = sr.squeeze().permute(1, 2, 0).float().cpu().clamp_(0, 1).numpy()
        # Save the sr image
        # Save the image in a dir with the same name
        img_name = lr_img_path[i].split('/')[-1]
        save_path = os.path.join(save_dir, img_name)
        plt.imsave(save_path, (sr*255).astype(np.uint8))
