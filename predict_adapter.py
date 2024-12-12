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
import warnings
import pandas as pd
from tqdm import tqdm

# Filter out all warnings
warnings.filterwarnings("ignore")



def calculate_adapter_avg_psnr(test_set, model, yadapt, scale, **save_opt):
    # Extract optional parameters or set defaults
    save_img = save_opt.get('save_img', False)
    save_name = save_opt.get('save_name', None)
    save_step = save_opt.get('save_step', 00)

    try:
        # Check the save path really exists
        assert os.path.exists(save_name)
        # Create a folder to save the images
        os.makedirs(os.path.join(save_name, save_step), exist_ok=True)
    except Exception as e:
        print(f"Not finding save path: {e}")
        
    model.eval()
    total_psnr = 0
    stride = test_set.pretrained_sam_img_size - test_set.overlap
    scale_factor = test_set.scale
    for iter, test_data in enumerate(test_set):
        batch_LR_image, HR_image, batch_yadapt_features = test_data[0], test_data[1], test_data[2]
        patches, (img_height, img_width), (padded_height, padded_width) = test_data[3], test_data[4], test_data[5]
        if yadapt == 'False':
            batch_yadapt_features = torch.zeros_like(batch_yadapt_features)
        batch_size = batch_LR_image.size(0)
        split_size = 1000  # Adjust this value based on your GPU memory capacity
        batch_Pre_image_list = []
        with torch.no_grad():
            for i in range(0, batch_size, split_size):
                batch_LR_image_split = batch_LR_image[i:i+split_size]
                batch_yadapt_features_split = batch_yadapt_features[i:i+split_size]

                # batch_Pre_image_split = model(batch_LR_image_split, batch_yadapt_features_split)
                batch_Pre_image_split = model(batch_LR_image_split)

                batch_Pre_image_list.append(batch_Pre_image_split)
        batch_Pre_image = torch.cat(batch_Pre_image_list, dim=0)
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

        if save_img:
            try:
                save_path = os.path.join(save_name, save_step, f'{iter}.png')
                Image.fromarray(super_res_image).save(save_path)
            except Exception as e:
                print(f"Error saving image: {e}")
                
        super_res_image = rgb2ycbcr(super_res_image, only_y=True)
        HR_image = rgb2ycbcr(HR_image, only_y=True)
        psnr = calculate_psnr(super_res_image, HR_image, border=scale)
        total_psnr += psnr
    avg_psnr = total_psnr / len(test_set)
    return avg_psnr

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

def load_model(model_path, model_config):
    with open(model_config, 'r') as file:
        config = yaml.safe_load(file)
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

    checkpoint = torch.load(model_path)

    model.eval()
    model.cuda()
    model = torch.nn.DataParallel(model) 


    if yadapt == 'False':
        checkpoint = {k: v for k, v in checkpoint.items() if 'module.self_attention' not in k}
    
    model.load_state_dict(checkpoint,strict=True)
    print('Resume from checkpoint from {}'.format(model_path))
    return model

def main_loop(config_path, model, gpu_ids, yadapt):
    # if yadapt == 'True':
    #     print('Yadapt is True SwinIRAdapter')
    # else:
    #     print('Yadapt is False Using zero yadapt as training SwinIR')

    # print('Config path: {}'.format(config_path))

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    scale = config['test']['scale']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
    test_set = SuperResolutionYadaptDataset(config=config['test'])
    avg_psnr = calculate_adapter_avg_psnr(test_set, model, yadapt, scale)
    # print('Avg PSNR: {:.2f}'.format(avg_psnr))
    return avg_psnr

def create_csv(config_list, avg_psnrs, model_path, current_dir):
    model_step = os.path.basename(model_path).split('_')[0]
    # Dataset Name, PSNR
    csv_path = os.path.join(current_dir, 'test_result_{}.csv'.format(model_step))
    
    config_name = [os.path.basename(config_path) for config_path in config_list]
    config_name = [config_name.replace('.yaml', '') for config_name in config_name]

    # Use pandas to create a csv file
    df = pd.DataFrame({'Dataset Name': config_name, 'PSNR': avg_psnrs})
    df.to_csv(csv_path, index=False)


def test_main(config_path_list, model_path, gpu_ids, yadapt, model_config):
    avg_psnrs = []
    model = load_model(model_path, model_config)
    current_dir = os.getcwd()
    for config_path in tqdm(config_path_list):
        avg_psnr = main_loop(config_path, model, gpu_ids, yadapt)
        # 保留两位小数
        avg_psnrs.append(round(avg_psnr, 2))
    create_csv(config_path_list, avg_psnrs, model_path, current_dir)

if __name__ == '__main__':
    import sys
    sys.path.append("/home/mayanze/PycharmProjects/SwinTF/")

    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

    # config_path_list = [
    #     '/home/mayanze/PycharmProjects/SwinTF/config/test_config/set5.yaml',
    #     '/home/mayanze/PycharmProjects/SwinTF/config/test_config/Set14test.yaml',
    #     '/home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml',
    #     '/home/mayanze/PycharmProjects/SwinTF/config/test_config/manga109test.yaml',
    #     '/home/mayanze/PycharmProjects/SwinTF/config/test_config/BSDS100.yaml',
    # ]

    config_path = '/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/noise/noise_sigma_0_150_general.yaml'
    config_path_list = [config_path]

    model_path = '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20241204110254/500000_model.pth'
    model_paths = [model_path]
    
    # model_paths = [
    #     '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240803080852/525000_model.pth',
    #     '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240803080852/520000_model.pth',
    #     '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240803080852/455000_model.pth',
    #     '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240803080852/450000_model.pth',
    #     '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240803080852/445000_model.pth',
    #     '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240803080852/440000_model.pth',
    # ]
    gpu_ids = '4,5'
    yadapt = 'True'

    model_config = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/scale2_base.yaml'

    for model_path in model_paths:
        test_main(config_path_list, model_path, gpu_ids, yadapt, model_config)

    # config_path, model_path, gpu_ids, yadapt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
   
    #DEBUG
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109test.yaml'
    # model_path= '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240803080852/500000_model.pth'
    # gpu_ids='4,5'
    # yadapt='True'





    # if yadapt == 'True':
    #     print('Yadapt is True SwinIRAdapter')
    # else:
    #     print('Yadapt is False Using zero yadapt as training SwinIR')

    # print('Config path: {}'.format(config_path))

    # with open(config_path, 'r') as file:
    #     config = yaml.safe_load(file)

    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)


    # # SwinIR+SAM
    # # model_path = 'experiments/SwinIR_20240317_192139/60000_model.pth'

    # scale = config['test']['scale']
    # # 3.1 SwinIR
    # model = SwinIRAdapter(upscale=config['network']['upsacle'], 
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
    #                 y_adapt_feature=torch.randn(1, 1, 1, 1)
    #                 )

    # # model = SwinIR(upscale=config['network']['upsacle'], 
    # #                 in_chans=config['network']['in_channels'],
    # #                 img_size=config['network']['image_size'],
    # #                 window_size=config['network']['window_size'],
    # #                 img_range=config['network']['image_range'],
    # #                 depths=config['network']['depths'],
    # #                 embed_dim=config['network']['embed_dim'],
    # #                 num_heads=config['network']['num_heads'],
    # #                 mlp_ratio=config['network']['mlp_ratio'],
    # #                 upsampler=config['network']['upsampler'],
    # #                 resi_connection=config['network']['resi_connection'],
    # #                 # y_adapt_feature=torch.randn(1, 1, 1, 1)
    # #                 )


    # # 加载预训练 SwinIR 模型
    # # model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
    # # # Use strict = True to check the model
    # # pretrained_model = torch.load(model_path)
    # # param_key_g = 'params'
    # # model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
    # checkpoint = torch.load(model_path)
    # # model.load_state_dict(checkpoint)




    # model.eval()
    # model.cuda()
    # model = torch.nn.DataParallel(model) 


    
    # if yadapt == 'False':
    #     # 纯SwinIR模型
    #     # 把checkpoint里面有关module.self_attention全部删除
    #     checkpoint = {k: v for k, v in checkpoint.items() if 'module.self_attention' not in k}
    
    
    
    
    
    # model.load_state_dict(checkpoint,strict=True)

    # print('Resume from checkpoint from {}'.format(model_path))

    # # if config['test']['precomputed']:
    # #     test_set = SuperResolutionPrecomputeYadaptDataset(config=config['test'])
    # # else:
    # #     test_set = SuperResolutionYadaptDataset(config=config['test'])
    
    # test_set = SuperResolutionYadaptDataset(config=config['test'])

    # # super_res_image = np.zeros((stride * (padded_height // stride) * scale_factor, stride * (padded_width // stride) * scale_factor), dtype=np.uint8)
    # # for (patch, x, y) in patches:
    # #     # x_pos = x * scale_factor
    # #     # y_pos = y * scale_factor
    # #     # 这样写的代价是还要裁掉周围一圈的overlap，等于在HR图像上丢失了一圈像素，但是这样我不会写了，所以还是分成三种情况
    # #     num_y = y // stride
    # #     num_x = x // stride




    # avg_psnr = calculate_adapter_avg_psnr(test_set, model, yadapt, scale)
    # print('Avg PSNR: {:.2f}'.format(avg_psnr))
