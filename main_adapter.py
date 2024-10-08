import os 
import sys
import yaml
import math
import torch 
import time
import wandb
import random
import numpy as np
import swanlab
from data.dataloader import SuperResolutionYadaptDataset
from torch.utils.data import DataLoader
from nets.swinir import SwinIRAdapter, SwinIR
from utils.utils_image import calculate_psnr, permute_squeeze
from utils.utils_data import print_config_as_table, load_config
from predict import evaluate_with_lrhr_pair, evaluate_with_hr
from predict_adapter import calculate_adapter_avg_psnr
import warnings

from main_test_swinir import test_main
# Filter out the specific warning
warnings.filterwarnings("ignore", message="Leaking Caffe2 thread-pool after fork. (function pthreadpool)")

# import argparse

# parser = argparse.ArgumentParser(description='Process some parameters.')
# parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
# parser.add_argument('--train_swinir', action='store_true', default=False, help='Train SwinIR model')
# args = parser.parse_args()
# DEBUG = args.debug
# train_swinir = args.train_swinir

DEBUG = True
train_swinir = False

print('Using train_swinir: {}'.format(train_swinir))

# from torch.utils.data._utils.collate import default_collate

# def custom_collate_fn(batch):
#     try:
#         return default_collate(batch)
#     except RuntimeError as e:
#         print(f"Collate error: {e}")
#         print(f"Batch size: {len(batch)}")
#         a = []
#         b = []
#         c = []
#         for i, items in enumerate(batch):
#             a.append
#                 print(f"Item {i} type: {type(item)}")
#                 # Print the shape of each item
#                 print(f"Item {i} shape: {item.shape}")
#         # Handle the error or skip the batch
#         return None

def process_config(config):
    config['train']['resume'] = config['train'].get('resume_optimizer') is not None and config['network'].get('resume_network') is not None

    # gpu_ids = config['train']['gpu_ids']

    # if train_swinir:
    #     config['train']['gpu_ids'] = [0,1,2,3]
    # else:
    #     config['train']['gpu_ids'] = [4,5,6,7]

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in config['train']['gpu_ids'])

    if torch.cuda.is_available():
        print('Using GPU: {}'.format(config['train']['gpu_ids']))

    if config['train']['resume']:
        seed = config['train']['seed']
        wandb_name = config['train']['wandb_name']
        wandb_id = config['train']['wandb_id']
        print('Using seed: {}'.format(seed))
        

    else:
        seed = random.randint(1, 10000)
        print('Random seed: {}'.format(seed))
        config['train']['seed'] = seed
        print('Using seed: {}'.format(seed))
        date = time.strftime('%m%d', time.localtime()) 
        wandb_name = f'{len(config["train"]["gpu_ids"])}卡SwinIRAdapter_{date}'
        detail_date = time.strftime('%Y%m%d%H%M%S', time.localtime())
        wandb_id = f'{detail_date}'
        config['train']['wandb_name'] = wandb_name
        config['train']['wandb_id'] = wandb_id
    
    config['train']['save_path'] = os.path.join(config['train']['save_path'], config['train']['type'] + '_' + config['train']['wandb_id'])
    # print('Save path: {}'.format(config['train']['save_path']))
    return config


# =================================================
# 0 Config，Global Parameters 部分
# =================================================
config_path = '/home/mayanze/PycharmProjects/SwinTF/config/set5_x4.yaml'
if DEBUG:
    config_path = '/home/mayanze/PycharmProjects/SwinTF/config/set5_debug.yaml'

config = load_config(config_path)
config = process_config(config)

if train_swinir:
    config['train']['seed'] = 2024
    config['train']['gpu_ids'] = [0,1,2,3]


# if torch.cuda.is_available():
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#     print('Using GPU: [0, 1, 2, 3]')
# else:
#     sys.exit('No GPU available')


random.seed(config['train']['seed'])
np.random.seed(config['train']['seed'])
torch.manual_seed(config['train']['seed'])
torch.cuda.manual_seed_all(config['train']['seed'])

if not DEBUG:
    if not os.path.exists(config['train']['save_path']):
        os.makedirs(config['train']['save_path'])

    with open(os.path.join(config['train']['save_path'], 'config.yaml'), 'w') as file:
        yaml.dump(config, file)


if not DEBUG:
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(
        project='SwinIR',
        id=config['train']['wandb_id'],
        name=config['train']['wandb_name'],
        config=config,
        resume='allow'
    )

    swanlab.init(
        project='SwinIR',
        experiment_name=config['train']['wandb_id'],
        description=config['train']['wandb_name'],
        config=config
    )


        
# =================================================
# 1 Dataset 部分
# =================================================
train_set = SuperResolutionYadaptDataset(config=config['train'])
test_set = SuperResolutionYadaptDataset(config=config['test'])



# ==================================================
# 2 Dataloader 部分
# ==================================================
train_size = int(math.ceil(len(train_set) / config['train']['batch_size']))
# 单机多卡，不用用 DistributedSampler
train_loader = DataLoader(train_set,
                          batch_size=config['train']['batch_size'],
                          shuffle=config['train']['shuffle'],
                          num_workers=config['train']['num_workers'],
                          drop_last=True,
                          pin_memory=True)

# train_loader = DataLoader(train_set,
#                           batch_size=32,
#                           shuffle=config['train']['shuffle'],
#                           num_workers=0,
#                           drop_last=True,
#                           pin_memory=True,
#                           collate_fn=custom_collate_fn)

test_loader = DataLoader(test_set,
                          batch_size=config['test']['batch_size'],
                          shuffle=config['test']['shuffle'],
                          num_workers=config['test']['num_workers'],
                          drop_last=False,
                          pin_memory=True)

# ==================================================
# 3 Network 部分
# ==================================================

# 3.1 SwinIR

if train_swinir:
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
else:
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

if config['network']['upsacle'] == 2:
    model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
elif config['network']['upsacle'] == 4:
    model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth'
else:
    raise ValueError(f"Invalid upsacle value: {config['network']['freeze_network']}")

# 加载预训练 SwinIR 模型
# Use strict = True to check the model
pretrained_model = torch.load(model_path)
param_key_g = 'params'
model.load_state_dict(pretrained_model[param_key_g], strict=False)
# ================================================
# 4 Loss, Optimizer 和 Scheduler 部分
# ================================================

if config['network']['freeze_network']:
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.self_attention.parameters():
        param.requires_grad = True
    
    for param in model.adapt_conv.parameters():
        param.requires_grad = True
    

# 4.1 Loss 定义
if config['train']['loss'] == 'l1':
    criterion = torch.nn.L1Loss()


# ==================================================
# 5 Training 部分
# ==================================================
# 5.1 训练初始化，找到 Step
device = torch.device('cuda' if config['train']['gpu_ids'] is not None else 'cpu')

model.train()
model.cuda()
model = torch.nn.DataParallel(model)

# 3.2 设计断点续训的机制
if config['network']['resume_network']:
    checkpoint = torch.load(config['network']['resume_network'])
    model.load_state_dict(checkpoint)
    print('Resume from checkpoint from {}'.format(config['network']['resume_network']))


# 4.2 Optimizer 定义
if config['train']['optimizer'] == 'adam':
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            print('Params [{:s}] will not optimize.'.format(k))

    if config['train']['resume_optimizer']:
        optimizer = torch.optim.Adam(optim_params,
                                    lr=config['train']['lr'],
                                    betas=(0.9, 0.999),
                                    weight_decay=config['train']['weight_decay'])
        checkpoint = torch.load(config['train']['resume_optimizer'], map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        optimizer.load_state_dict(checkpoint)
        print('Resume from optimizer from {}'.format(config['train']['resume_optimizer']))
    else:
        optimizer = torch.optim.Adam(optim_params,
                                    lr=config['train']['lr'],
                                    betas=(0.9, 0.999),
                                    weight_decay=config['train']['weight_decay'])

# 4.3 Scheduler 定义
if config['train']['scheduler'] == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=config['train']['milestones'], # milestones: [250000, 400000, 450000, 475000, 500000]
                                                    gamma=config['train']['gamma']) # gamma: 0.5


# 用 Step 来记录训练的次数
if config['network']['resume_network'] and config['train']['resume_optimizer']:
    # 步长保存为模型的名字 {step}_{model_name}.pth
    step_network = config['network']['resume_network'].split('/')[-1].split('.')[0].split('_')[0]
    # 步长保存为优化器的名字 {step}_{optimizer_name}.pth
    step_optimizer = config['train']['resume_optimizer'].split('/')[-1].split('.')[0].split('_')[0]
    assert step_network == step_optimizer, 'Network and Optimizer should have the same step.'
    current_step = int(step_network)

else:
    current_step = 0

print_config_as_table(config, config_path)

from tqdm import tqdm

# for epoch in range(10000000000):
#     for _, train_data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Progress"):
#         current_step += 1
#         # 5.3.1 数据准备
#         train_LR, train_HR, y_adapt = train_data

# 5.3 开始训练
smooth_loss = []
running_loss = 0.0
for epoch in range(10000000000):
    for _, train_data in enumerate(train_loader):
        current_step += 1
        # 5.3.1 数据准备
        train_LR, train_HR, y_adapt = train_data
        train_HR = train_HR.cuda()
        train_LR = train_LR.cuda()

        scheduler.step(current_step)

        if DEBUG:
            # DEBUG 这里将 HR，LR 都保存出来
            # 保存 HR
            from PIL import Image

            HR_image = train_HR.cpu().permute(0,2,3,1).numpy()
            HR_image = HR_image[0]
            HR_image = HR_image * 255
            HR_image = HR_image.astype(np.uint8)
            HR_image = Image.fromarray(HR_image)
            HR_image.save('HR_image.png')

            # 保存 LR
            LR_image = train_LR.cpu().permute(0,2,3,1).numpy()
            LR_image = LR_image[0]
            LR_image = LR_image * 255
            LR_image = LR_image.astype(np.uint8)
            LR_image = Image.fromarray(LR_image)
            LR_image.save('LR_image.png')

        y_adapt = y_adapt.cuda()
        optimizer.zero_grad()

        if train_swinir:
            output = model.forward(train_LR)
        else:
            output = model.forward(train_LR, y_adapt)

        if DEBUG:
            # DEBUG 保存 output
            output_img = output.clamp(0, 1).detach().cpu().permute(0,2,3,1).numpy()
            output_img = output_img[0]
            output_img = output_img * 255
            output_img = output_img.astype(np.uint8)
            output_img = Image.fromarray(output_img)
            output_img.save('output.png')

        loss = criterion(output, train_HR)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        smooth_loss.append(loss.item())
        if len(smooth_loss) > 100:
            smooth_loss.pop(0)
        loss_smooth = sum(smooth_loss) / len(smooth_loss)

        # 5.3.3 打印训练信息
        if current_step % config['train']['step_print'] == 0:
            print('Epoch: {:d}, Step: {:d}, Loss: {:.4f}, Smoothed Loss: {:.4f}, LR: {:.8f}'.format(epoch, current_step, loss.item(), loss_smooth, scheduler.get_last_lr()[0]))
            wandb.log({"Epoch": epoch, "Step": current_step, "Loss": loss.item(), "Smoothed Loss": loss_smooth, "Learning Rate": scheduler.get_last_lr()[0]})
            swanlab.log({"Epoch": epoch, "Step": current_step, "Loss": loss.item(), "Smoothed Loss": loss_smooth, "Learning Rate": scheduler.get_last_lr()[0]})
        # 5.3.4 保存模型
        if current_step % config['train']['step_save'] == 0:
            torch.save(model.state_dict(), os.path.join(config['train']['save_path'], '{:d}_model.pth'.format(current_step)))
            torch.save(optimizer.state_dict(), os.path.join(config['train']['save_path'], '{:d}_optimizer.pth'.format(current_step)))
        # 5.3.5 测试模型
        if current_step % config['train']['step_test'] == 0:
            model.eval()
            if train_swinir:
                if config['test']['test_LR'] == "BIC":
                    avg_psnr = evaluate_with_hr(config['test']['test_HR'], model, config['train']['scale'])
                else:
                    avg_psnr = evaluate_with_lrhr_pair(config['test']['test_HR'], config['test']['test_LR'], model, config['train']['scale'])
                print('Epoch: {:d}, Step: {:d}, Avg PSNR: {:.4f}'.format(epoch, current_step, avg_psnr))
                wandb.log({"Epoch": epoch, "Step": current_step, "Avg PSNR": avg_psnr})
                swanlab.log({"Epoch": epoch, "Step": current_step, "Avg PSNR": avg_psnr})
            else:
                if config['test']['test_LR'] == "BIC":
                    raise ValueError("BIC is not supported for SwinIRAdapter")
                else:
                    # avg_psnr = calculate_adapter_avg_psnr(test_set, model, yadapt=True, scale=config['train']['scale'])
                    avg_psnr = test_main(config_path, model, test_swinir=False)
                print('Epoch: {:d}, Step: {:d}, Avg PSNR: {:.4f}'.format(epoch, current_step, avg_psnr))
                wandb.log({"Epoch": epoch, "Step": current_step, "Avg PSNR": avg_psnr})
                swanlab.log({"Epoch": epoch, "Step": current_step, "Avg PSNR": avg_psnr})
            model.train()