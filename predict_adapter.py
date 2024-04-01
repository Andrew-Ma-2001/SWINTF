# 1 Load in model
# 2 Predict the input image list
# 3 Calculate the PSNR

import os
from PIL import Image
import torch
import numpy as np
from utils.utils_image import permute_squeeze, calculate_psnr, imresize_np
from nets.swinir import SwinIRAdapter, SwinIR
from data.dataloader import SuperResolutionYadaptDataset
import yaml
import matplotlib.pyplot as plt



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

if __name__ == '__main__':
    import sys
    sys.path.append("/home/mayanze/PycharmProjects/SwinTF/")
    config_path, model_path, gpu_ids, yadapt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml'

    if yadapt == 'True':
        print('Yadapt is True SwinIRAdapter')
    else:
        print('Yadapt is False Using zero yadapt as training SwinIR')

    print('Config path: {}'.format(config_path))

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)


    # SwinIR+SAM
    # model_path = 'experiments/SwinIR_20240317_192139/60000_model.pth'

    scale = config['test']['scale']
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

    # model = SwinIR(upscale=config['network']['upsacle'], 
    #                 in_chans=config['network']['in_channels'],
    #                 img_size=config['network']['image_size'],
    #                 window_size=config['network']['window_size'],
    #                 img_range=config['network']['image_range'],
    #                 depths=config['network']['depths'],
    #                 embed_dim=config['network']['embed_dim'],
    #                 num_heads=config['network']['num_heads'],
    #                 mlp_ratio=config['network']['mlp_ratio'],
    #                 upsampler=config['network']['upsampler'],
    #                 resi_connection=config['network']['resi_connection'],
    #                 # y_adapt_feature=torch.randn(1, 1, 1, 1)
    #                 )


    # 加载预训练 SwinIR 模型
    # model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
    # # Use strict = True to check the model
    # pretrained_model = torch.load(model_path)
    # param_key_g = 'params'
    # model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
    checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint)




    model.eval()
    model.cuda()
    model = torch.nn.DataParallel(model) 


    
    if yadapt == 'False':
        # 纯SwinIR模型
        # 把checkpoint里面有关module.self_attention全部删除
        checkpoint = {k: v for k, v in checkpoint.items() if 'module.self_attention' not in k}
    
    
    
    
    
    model.load_state_dict(checkpoint,strict=False)

    print('Resume from checkpoint from {}'.format(model_path))

    # if config['test']['precomputed']:
    #     test_set = SuperResolutionPrecomputeYadaptDataset(config=config['test'])
    # else:
    #     test_set = SuperResolutionYadaptDataset(config=config['test'])
    
    test_set = SuperResolutionYadaptDataset(config=config['test'])

    # super_res_image = np.zeros((stride * (padded_height // stride) * scale_factor, stride * (padded_width // stride) * scale_factor), dtype=np.uint8)
    # for (patch, x, y) in patches:
    #     # x_pos = x * scale_factor
    #     # y_pos = y * scale_factor
    #     # 这样写的代价是还要裁掉周围一圈的overlap，等于在HR图像上丢失了一圈像素，但是这样我不会写了，所以还是分成三种情况
    #     num_y = y // stride
    #     num_x = x // stride


    total_psnr = 0
    stride = test_set.pretrained_sam_img_size - test_set.overlap
    scale_factor = test_set.scale

    for iter, test_data in enumerate(test_set):
        batch_LR_image, HR_image, batch_yadapt_features = test_data[0], test_data[1], test_data[2]
        patches, (img_height, img_width), (padded_height, padded_width) = test_data[3], test_data[4], test_data[5]
        

        if yadapt == 'False':
            batch_yadapt_features = torch.zeros_like(batch_yadapt_features)

        with torch.no_grad():
            batch_Pre_image = model(batch_LR_image, batch_yadapt_features)

        # with torch.no_grad():
        #     batch_Pre_image = model(batch_LR_image)

        # batch_Pre_image 的形状  [x*y, 3, 48*scale, 48*scale] -> [x, y, 3,  48*scale, 48*scale] -> [48*x*scale, 48*y*scale, 3] 
        # batch_LR_image = LR_image.reshape(x//48, 48, y//48, 48, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 3, 48, 48)
        # batch_Pre_image = batch_Pre_image.view(x//48, y//48, 3, 48*scale, 48*scale)
        # batch_Pre_image = batch_Pre_image.permute(0, 3, 1, 4, 2).contiguous().view(x*scale, y*scale, 3)
        # batch_Pre_image = batch_Pre_image.clamp(0, 1)
        # # Change to numpy 
        # batch_Pre_image = batch_Pre_image.cpu().detach().numpy()
        # batch_Pre_image = batch_Pre_image * 255
        # HR_image = HR_image * 255

        batch_Pre_image = batch_Pre_image.clamp(0, 1).cpu().detach().permute(0,2,3,1).numpy()
        batch_Pre_image = batch_Pre_image * 255
        batch_Pre_image = batch_Pre_image.astype(np.uint8)
        super_res_image = np.zeros((stride * (padded_height // stride) * scale_factor, stride * (padded_width // stride) * scale_factor, 3), dtype=np.uint8)

        for (patch, x, y), pre_image in zip(patches, batch_Pre_image):
            num_y = y // stride
            num_x = x // stride
            patch = pre_image[(test_set.overlap//2)*scale_factor:(test_set.pretrained_sam_img_size-test_set.overlap//2)*scale_factor, (test_set.overlap//2)*scale_factor:(test_set.pretrained_sam_img_size-test_set.overlap//2)*scale_factor, :]
            super_res_image[num_y*stride*scale_factor:(num_y+1)*stride*scale_factor, num_x*stride*scale_factor:(num_x+1)*stride*scale_factor,:] = patch

        super_res_image = super_res_image[:img_height*scale_factor-test_set.overlap*scale_factor, :img_width*scale_factor - test_set.overlap*scale_factor]


        plt.imsave('{}.png'.format(iter), super_res_image.astype(np.uint8))
        # print('Save {}.png'.format(iter))
        plt.imsave('{}_HR.png'.format(iter), HR_image.astype(np.uint8))

        super_res_image = rgb2ycbcr(super_res_image, only_y=True)
        HR_image = rgb2ycbcr(HR_image, only_y=True)

        psnr = calculate_psnr(super_res_image, HR_image, border=scale)
        total_psnr += psnr
        # print('PSNR: {:.2f}'.format(psnr))

    print('Avg PSNR: {:.2f}'.format(total_psnr / len(test_set)))
