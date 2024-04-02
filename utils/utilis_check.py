import sys
import yaml
import os
import cv2
import torch
sys.path.append("/home/mayanze/PycharmProjects/SwinTF/")

def check_train_pair():
    from data.dataloader import SuperResolutionYadaptDataset
    config = '/home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml'
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    train_set = SuperResolutionYadaptDataset(config=cfg['train'])
    
    LR_image, HR_image, _ = train_set.__getitem__(0)
    LR_image = LR_image.permute(1, 2, 0).numpy() * 255
    HR_image = HR_image.permute(1, 2, 0).numpy() * 255
    
    cv2.imwrite('LR_image.png', LR_image)
    cv2.imwrite('HR_image.png', HR_image)

def check_model_with_pretrain_model():
    from nets.swinir import SwinIRAdapter
    config_path = '/home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml'
    print('Config path: {}'.format(config_path))
    # print('Using zero yadapt')
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/Set14test.y
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    gpu_ids = config['train']['gpu_ids']
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    # SwinIR+SAM
    # model_path = '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_
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

    model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'

    # Use strict = True to check the model
    pretrained_model = torch.load(model_path)
    param_key_g = 'params'
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=False)
# model.load_state_dict(torch.load(model_path), strict=True)

def check_train_yadapt_shape():
    # load the npy file
    import numpy as np
    file_path = 'dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic/X2_yadapt/0824x2_yadapt.npy'
    yadapt_feature = np.load(file_path)


def check_train_pair():
    from data.dataloader import SuperResolutionYadaptDataset
    config_path = '/home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    train_set = SuperResolutionYadaptDataset(config=config['train'])
    LR_image, HR_image, yadapt_feature = train_set.__getitem__(0)
    print(LR_image.shape, HR_image.shape, yadapt_feature.shape)


def check_config_consistency(config):
    # Define the expected keys and their types
    expected_keys = {
        'train': dict,
        'test': dict,
        'network': dict
    }

    # Check if all expected keys are present
    for key, expected_type in expected_keys.items():
        if key not in config:
            raise Exception(f"Missing key in config: {key}")
        if not isinstance(config[key], expected_type):
            raise Exception(f"Expected type {expected_type} for key {key}, but got {type(config[key])}")

    # Check the keys in the 'train', 'test', and 'network' sections
    train_keys = ['type', 'mode', 'scale', 'patch_size', 'train_HR', 'train_LR', 'batch_size', 'shuffle', 'num_workers', 'gpu_ids', 'loss', 'optimizer', 'lr', 'weight_decay', 'resume_optimizer', 'step_save', 'step_test', 'step_print', 'scheduler', 'milestones', 'gamma', 'save_path']
    test_keys = ['mode', 'scale', 'patch_size', 'test_HR', 'test_LR', 'batch_size', 'shuffle', 'num_workers']
    network_keys = ['upsacle', 'in_channels', 'image_size', 'window_size', 'image_range', 'depths', 'embed_dim', 'num_heads', 'mlp_ratio', 'upsampler', 'resi_connection', 'resume_network']

    for key in train_keys:
        if key not in config['train']:
            raise Exception(f"Missing key in config['train']: {key}")

    for key in test_keys:
        if key not in config['test']:
            raise Exception(f"Missing key in config['test']: {key}")

    for key in network_keys:
        if key not in config['network']:
            raise Exception(f"Missing key in config['network']: {key}")

    # New rules
    # 1. The batch size on each gpu should be 8
    if config['train']['batch_size'] / len(config['train']['gpu_ids']) != 8:
        print("Warning: The batch size on each GPU should be 8.")

    # 2. Train scale and test scale should be the same
    if config['train']['scale'] != config['test']['scale']:
        print("Warning: Train scale and test scale should be the same.")

    # 3. The patch size should be divisible by the scale
    if config['train']['patch_size'] % config['train']['scale'] != 0:
        print("Warning: The patch size should be divisible by the scale.")

    # 4. The resume_network and resume_optimizer should have the same step
    if config['train']['resume_optimizer'] != config['network']['resume_network']:
        print("Warning: The resume_network and resume_optimizer should have the same step.")

    # 4.1 It should either be None or an existing file
    if config['train']['resume_optimizer'] != 'None' and not os.path.isfile(config['train']['resume_optimizer']):
        print("Warning: resume_optimizer should either be None or an existing file.")
    if config['network']['resume_network'] != 'None' and not os.path.isfile(config['network']['resume_network']):
        print("Warning: resume_network should either be None or an existing file.")


    print("Config is consistent.")
