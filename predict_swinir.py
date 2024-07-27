# 这个文件的目的是给 Swinir 写一个通用的检测流程
from predict import evaluate_with_lrhr_pair, evaluate_with_hr
import torch
from nets.swinir import SwinIR
import yaml
import os
import sys
sys.path.append("/home/mayanze/PycharmProjects/SwinTF/")


config_path_list = [
    '/home/mayanze/PycharmProjects/SwinTF/config/Set5.yaml',
    '/home/mayanze/PycharmProjects/SwinTF/config/urban100test.yaml',
    '/home/mayanze/PycharmProjects/SwinTF/config/manga109test.yaml',
    '/home/mayanze/PycharmProjects/SwinTF/config/BSDS100.yaml',
    '/home/mayanze/PycharmProjects/SwinTF/config/Set14test.yaml'
]
# 还是用 yaml 控制
# config_path = sys.argv[1]
# config_path = '/home/mayanze/PycharmProjects/SwinTF/config/urban100test.yaml'

config_path = config_path_list[0]
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

gpu_ids = config['train']['gpu_ids']
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

model_path = '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240711002251/50000_model.pth'
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

# param_key_g = 'params'
# model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

model = torch.nn.DataParallel(model) 
model.load_state_dict(pretrained_model)

print('Resume from checkpoint from {}'.format(model_path))

for config_path in config_path_list:
    print(f'config_path: {config_path}')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if config['test']['test_LR'] == "BIC":
        avg_psnr = evaluate_with_hr(config['test']['test_HR'], model, scale)
    else:
        avg_psnr = evaluate_with_lrhr_pair(config['test']['test_HR'], config['test']['test_LR'], model, scale)


# Example Command:
# python predict_swinir.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml
# python predict_swinir.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/manga109test.yaml