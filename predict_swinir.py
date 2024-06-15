# 这个文件的目的是给 Swinir 写一个通用的检测流程
from predict import evaluate_with_lrhr_pair
import torch
from nets.swinir import SwinIR
import yaml
import os
import sys
sys.path.append("/home/mayanze/PycharmProjects/SwinTF/")



# 还是用 yaml 控制
config_path = sys.argv[1]
# config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/blur_aniso.yaml'
print(f'config_path: {config_path}')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

gpu_ids = config['train']['gpu_ids']
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

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

pretrained_model = torch.load(model_path)
param_key_g = 'params'
model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
model = torch.nn.DataParallel(model) 

print('Resume from checkpoint from {}'.format(model_path))

avg_psnr = evaluate_with_lrhr_pair(config['test']['test_HR'], config['test']['test_LR'], model, scale)
# print(f'avg_psnr: {avg_psnr}')