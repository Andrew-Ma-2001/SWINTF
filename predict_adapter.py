# 1 Load in model
# 2 Predict the input image list
# 3 Calculate the PSNR

import os
from PIL import Image
import torch
import numpy as np
from utils.utils_image import permute_squeeze, calculate_psnr, imresize_np
from nets.swinir import SwinIRAdapter
from data.dataloader import SuperResolutionYadaptDataset
import yaml
import matplotlib.pyplot as plt


SAVE_NPY = False



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

def save_data(data, path):
    # Change the data from torch to numpy
    data = data.cpu().detach().numpy()
    np.save(path, data)

if __name__ == '__main__':
    config_path = 'config/example copy.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    gpu_ids = config['train']['gpu_ids']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    # Use one GPU to predict
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    # SwinIR+SAM
    model_path = '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240204_022316/295000_model.pth'

    scale = config['train']['scale']
    # 3.1 SwinIR
    model = SwinIRAdapter(upscale=config['network']['upsacle'], 
                    in_chans=config['network']['in_channels'],
                    img_size=config['network']['image_size'],
                    window_size=config['network']['window_size'],
                    img_range=config['network']['image_range'],
                    depths=config['network']['depths'],
                    embed_dim=config['network']['embed_dim'],
                    num_heads=config['network']['num_heads'],
                    mlp_ratio=config['network']['mlp_ratio'],
                    upsampler=config['network']['upsampler'],
                    resi_connection=config['network']['resi_connection'],
                    y_adapt_feature=torch.randn(1, 1, 1, 1)
                    )


    model.eval()
    model.cuda()
    model = torch.nn.DataParallel(model) 
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print('Resume from checkpoint from {}'.format(model_path))

    test_set = SuperResolutionYadaptDataset(config=config['test'])

    i = 0 
    total_psnr = 0
    for iter, test_data in enumerate(test_set):
        batch_LR_image, HR_image, batch_yadapt_features, (x, y) = test_data[0], test_data[1], test_data[2], test_data[3]
        


        batch_Pre_image = model(batch_LR_image, batch_yadapt_features)

        # Save the batch_Pre_image in numpy
        if SAVE_NPY:
            save_data(batch_Pre_image, 'batch_Pre_image_{}.npy'.format(i))
            save_data(batch_LR_image, 'batch_LR_image_{}.npy'.format(i))
            save_data(batch_yadapt_features, 'batch_yadapt_features_{}.npy'.format(i))

        #=====================================================================
        ### 1 这里PSNR每次运行的结果不一样，因为yadapt_features每次不一样，下面检查为什么不一样
        ### 2 
        # ====================================================================


        # batch_Pre_image 的形状  [x*y, 3, 48*scale, 48*scale] -> [x, y, 3,  48*scale, 48*scale] -> [48*x*scale, 48*y*scale, 3] 
        # batch_LR_image = LR_image.reshape(x//48, 48, y//48, 48, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 3, 48, 48)
        batch_Pre_image = batch_Pre_image.view(x//48, y//48, 3, 48*scale, 48*scale)
        batch_Pre_image = batch_Pre_image.permute(0, 3, 1, 4, 2).contiguous().view(x*scale, y*scale, 3)
        batch_Pre_image = batch_Pre_image.clamp(0, 1)
        # Change to numpy 
        batch_Pre_image = batch_Pre_image.cpu().detach().numpy()
        batch_Pre_image = batch_Pre_image * 255
        HR_image = HR_image * 255

        plt.imsave('{}.png'.format(i), batch_Pre_image.astype(np.uint8))
        print('Save {}.png'.format(i))
        i += 1


        psnr = calculate_psnr(batch_Pre_image, HR_image, border=scale)
        total_psnr += psnr
        print('PSNR: {:.2f}'.format(psnr))

        break
    print('Avg PSNR: {:.2f}'.format(total_psnr / len(test_set)))


    # gt_path = 'dataset/testsets/Set14/GTmod12'
    # lr_path = 'dataset/testsets/Set14/LRbicx4'
    # # scale = 4
    # evaluate_with_lrhr_pair(gt_path, lr_path, model)

    # gt_path = 'dataset/testsets/urban100'
    # evaluate_with_hr(gt_path, model)