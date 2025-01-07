import os 
import sys
import yaml
import math
import torch 
import time
import wandb
import random
import numpy as np
from data.dataloader import SuperResolutionYadaptDataset
# from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from nets.swinir import SwinIR
from utils.utils_dist import *
from utils.utils_data import print_config_as_table, load_config
from predict import evaluate_with_lrhr_pair, evaluate_with_hr

import warnings
from main_test_swinir import test_main
# Filter out the specific warning
warnings.filterwarnings("ignore", message="Leaking Caffe2 thread-pool after fork. (function pthreadpool)")
import requests

import argparse

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
parser.add_argument('--train_swinir', action='store_true', default=False, help='Train SwinIR model')
# parser.add_argument('--launcher', default='pytorch', help='job launcher')
# parser.add_argument('--dist', default='store_true')
parser.add_argument('--local-rank', type=int, default=0)
parser.add_argument('--mode', type=int, default=1, help='Mode for training')
parser.add_argument('--swinir_mode', type=str, default='swinir', help='Mode for SwinIR model')
parser.add_argument('--config', type=str, required=True, help='Path to config file')
parser.add_argument('--gpu', type=str, default='0', help='GPU id(s) to use (comma-separated, e.g., "0,1,2,3")')

args = parser.parse_args()


DEBUG = args.debug
train_swinir = args.train_swinir

mode = args.mode # TODO: 修改 mode
swinir_mode = args.swinir_mode
config_path = args.config
gpus = args.gpu

if swinir_mode == 'swinir':
    from nets.swinir import SwinIRAdapter
elif swinir_mode == 'strong_norm':
    from nets.swinir_strongnorm import SwinIRStrongNorm as SwinIRAdapter
elif swinir_mode == 'pixelshuffle':
    from nets.swinir_pixelshuffel import SwinIRPixelShuffel as SwinIRAdapter
elif swinir_mode == 'newfeature':
    from nets.swinir_newfeature import SwinIRNewFeature as SwinIRAdapter

print('Using train_swinir: {}'.format(train_swinir))

    
def process_config(config):
    config['train']['resume'] = config['train'].get('resume_optimizer') is not None and config['network'].get('resume_network') is not None
    config['train']['gpu_ids'] = [int(x.strip()) for x in gpus.split(',')]
    config['train']['swinir_mode'] = swinir_mode
    config['test']['swinir_mode'] = swinir_mode
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in config['train']['gpu_ids'])

    if torch.cuda.is_available():
        print('Using GPU: {}'.format(config['train']['gpu_ids']))
    else:
        sys.exit('No GPU available')

    if config['train']['resume']:
        seed = config['train']['seed']
        wandb_name = config['train']['wandb_name']
        wandb_id = config['train']['wandb_id']
        print('Using seed: {}'.format(seed))

    else:
        if train_swinir:
            seed = 2024
        else:
            seed = random.randint(1, 1000000)
        print('Random seed: {}'.format(seed))
        config['train']['seed'] = seed
        config['test']['seed'] = seed
        print('Using seed: {}'.format(seed))
        date = time.strftime('%m%d', time.localtime()) 
        wandb_name = f'{len(config["train"]["gpu_ids"])}卡SA_{date}' + '_' + config['train']['swinir_mode']
        detail_date = time.strftime('%Y%m%d%H%M%S', time.localtime())
        wandb_id = f'{detail_date}'
        config['train']['wandb_name'] = wandb_name
        config['train']['wandb_id'] = wandb_id
    
    config['train']['save_path'] = os.path.join(config['train']['save_path'], config['train']['type'] + '_' + config['train']['wandb_id'])
    return config



# @record
def main():
    # =================================================
    # 0 Config，Global Parameters 部分
    # =================================================

    # train_swinir = True

    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/set5_mode1.yaml'
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/set5_mode1_p192.yaml'
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/set5_mode3.yaml'
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/Set5_ddp.yaml'
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/set5_x4.yaml'

    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/set5_debug.yaml'
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/set5_freeze.yaml'



    config = load_config(config_path)
    config = process_config(config)

    config['train']['dist'] = True
    config['network']['mode'] = mode
    if 'best_model' in config['train']:
        config['train']['best_model'] = True
    else:
        config['train']['best_model'] = False

    if config['train']['dist']:
        # init_dist('pytorch')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier()

    config['train']['rank'], config['train']['world_size'] = get_dist_info()

    if config['train']['rank'] == 0:
        if not os.path.exists(config['train']['save_path']):
            os.makedirs(config['train']['save_path'])

        with open(os.path.join(config['train']['save_path'], 'config.yaml'), 'w') as file:
            yaml.dump(config, file)

        if not DEBUG:
            # Test connection to wandb server
            try:
                # Attempt to connect with a 10 second timeout
                requests.get('https://api.wandb.ai', timeout=10)
                wandb_mode = 'online'
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                print('Could not connect to wandb server, running in offline mode')
                wandb_mode = 'offline'
                os.environ['WANDB_MODE'] = 'offline'

            if wandb_mode == 'online':
                wandb.init(
                    project='SwinIR',
                    id=config['train']['wandb_id'], 
                    name=config['train']['wandb_name'],
                    config=config,
                    resume='allow'
                )
            elif wandb_mode == 'offline':
                wandb.init(
                    project='SwinIR',
                    id=config['train']['wandb_id'], 
                    name=config['train']['wandb_name'],
                    config=config,
                    resume='allow',
                    dir='wandb_offline/'
                )
    if train_swinir:
        config['train']['seed'] = 2024
        config['test']['seed'] = 2024

    random.seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    torch.manual_seed(config['train']['seed'])
    torch.cuda.manual_seed_all(config['train']['seed'])

    # =================================================
    # 1 Dataset 部分
    # =================================================
    train_set = SuperResolutionYadaptDataset(config=config['train'])
    test_set = SuperResolutionYadaptDataset(config=config['test'])

    # ==================================================
    # 2 Dataloader 部分
    # ==================================================
    train_size = int(math.ceil(len(train_set) / config['train']['batch_size']))

    if config['train']['dist']:
        train_sampler = DistributedSampler(train_set, shuffle=config['train']['shuffle'], drop_last=True, seed=config['train']['seed'])
        train_loader = DataLoader(train_set,
                                  batch_size=config['train']['batch_size'] // len(config['train']['gpu_ids']),
                                  shuffle=False,
                                  num_workers=config['train']['num_workers'] // len(config['train']['gpu_ids']),
                                  drop_last=True,
                                  pin_memory=True,
                                  sampler=train_sampler)
        test_sampler = DistributedSampler(test_set, shuffle=config['test']['shuffle'], drop_last=True, seed=config['test']['seed'])
        test_loader = DataLoader(test_set,
                                  batch_size=config['test']['batch_size'],
                                  shuffle=False,
                                  num_workers=config['test']['num_workers'],
                                  drop_last=True,
                                  pin_memory=True,
                                  sampler=test_sampler)

    else:
        train_loader = DataLoader(train_set,
                                  batch_size=config['train']['batch_size'],
                                  shuffle=config['train']['shuffle'],
                                  num_workers=config['train']['num_workers'],
                                  drop_last=True,
                                  pin_memory=True)

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

    # 加载预训练 SwinIR 模型
    if config['network']['upsacle'] == 2:
        model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
    elif config['network']['upsacle'] == 4:
        model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth'


    # Use strict = True to check the model
    pretrained_model = torch.load(model_path)
    param_key_g = 'params'
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=False)
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

    if config['train']['dist']:
        find_unused_parameters = True
        use_static_graph = False
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(args.local_rank),
                                                        # device_ids=[torch.cuda.current_device()],
                                                        device_ids=[args.local_rank], output_device=args.local_rank,
                                                        find_unused_parameters=find_unused_parameters)
        if use_static_graph:
            model._set_static_graph()
    else:
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

    if config['train']['rank'] == 0:
        print_config_as_table(config, config_path)

    # 5.3 开始训练
    smooth_loss = []
    running_loss = 0.0
    best_loss = float('inf')
    for epoch in range(10000000000):
        if config['train']['dist']:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        for _, train_data in enumerate(train_loader):
            current_step += 1
            # 5.3.1 数据准备
            train_LR, train_HR, y_adapt = train_data
            train_HR = train_HR.cuda()
            train_LR = train_LR.cuda()

            scheduler.step(current_step)

            # if DEBUG:
            #     # DEBUG 这里将 HR，LR 都保存出来
            #     # 保存 HR
            #     from PIL import Image

            #     HR_image = train_HR.cpu().permute(0,2,3,1).numpy()
            #     HR_image = HR_image[0]
            #     HR_image = HR_image * 255
            #     HR_image = HR_image.astype(np.uint8)
            #     HR_image = Image.fromarray(HR_image)
            #     HR_image.save('HR_image.png')

            #     # 保存 LR
            #     LR_image = train_LR.cpu().permute(0,2,3,1).numpy()
            #     LR_image = LR_image[0]
            #     LR_image = LR_image * 255
            #     LR_image = LR_image.astype(np.uint8)
            #     LR_image = Image.fromarray(LR_image)
            #     LR_image.save('LR_image.png')

            y_adapt = y_adapt.cuda()

            optimizer.zero_grad()

            if train_swinir:
                output = model.forward(train_LR)
            else:
                output = model.forward(train_LR, y_adapt)

            # if DEBUG:
            #     # DEBUG 保存 output
            #     output_img = output.clamp(0, 1).detach().cpu().permute(0,2,3,1).numpy()
            #     output_img = output_img[0]
            #     output_img = output_img * 255
            #     output_img = output_img.astype(np.uint8)
            #     output_img = Image.fromarray(output_img)
            #     output_img.save('output.png')

            loss = criterion(output, train_HR)
            loss.backward()
            # clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
            smooth_loss.append(loss.item())
            if len(smooth_loss) > 100:
                smooth_loss.pop(0)
            loss_smooth = sum(smooth_loss) / len(smooth_loss)

            if not config['train']['best_model']:
                # 5.3.3 打印训练信息
                if current_step % config['train']['step_print'] == 0 and config['train']['rank'] == 0:
                    print('Epoch: {:d}, Step: {:d}, Loss: {:.4f}, Smoothed Loss: {:.4f}, LR: {:.8f}'.format(epoch, current_step, loss.item(), loss_smooth, scheduler.get_last_lr()[0]))
                    wandb.log({"Epoch": epoch, "Step": current_step, "Loss": loss.item(), "Smoothed Loss": loss_smooth, "Learning Rate": scheduler.get_last_lr()[0]})

                # 5.3.4 保存模型
                if current_step % config['train']['step_save'] == 0 and config['train']['rank'] == 0:
                    torch.save(model.state_dict(), os.path.join(config['train']['save_path'], '{:d}_model.pth'.format(current_step)))
                    torch.save(optimizer.state_dict(), os.path.join(config['train']['save_path'], '{:d}_optimizer.pth'.format(current_step)))

                # 5.3.5 测试模型
                if current_step % config['train']['step_test'] == 0 and config['train']['rank'] == 0:
                    model.eval()
                    if train_swinir:
                        if config['test']['test_LR'] == "BIC":
                            avg_psnr = evaluate_with_hr(config['test']['test_HR'], model, config['train']['scale'])
                        else:
                            avg_psnr = evaluate_with_lrhr_pair(config['test']['test_HR'], config['test']['test_LR'], model, config['train']['scale'])
                        print('Epoch: {:d}, Step: {:d}, Avg PSNR: {:.4f}'.format(epoch, current_step, avg_psnr))
                        wandb.log({"Epoch": epoch, "Step": current_step, "Avg PSNR": avg_psnr})
                    else:
                        if config['test']['test_LR'] == "BIC":
                            raise ValueError("BIC is not supported for SwinIRAdapter")
                        else:
                            avg_psnr = test_main(config_path, model, test_swinir=False)
                        print('Epoch: {:d}, Step: {:d}, Avg PSNR: {:.4f}'.format(epoch, current_step, avg_psnr))
                        wandb.log({"Epoch": epoch, "Step": current_step, "Avg PSNR": avg_psnr})
                    model.train()
            else:
                if current_step % config['train']['step_print'] == 0 and config['train']['rank'] == 0:
                    print('Epoch: {:d}, Step: {:d}, Loss: {:.4f}, Smoothed Loss: {:.4f}, LR: {:.8f}'.format(epoch, current_step, loss.item(), loss_smooth, scheduler.get_last_lr()[0]))
                if loss.item() < best_loss and config['train']['rank'] == 0:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), os.path.join(config['train']['save_path'], 'best_model.pth'))
                    print('Best loss: {:.4f}'.format(best_loss))
                    print('Best model saved to {}'.format(os.path.join(config['train']['save_path'], 'best_model.pth')))

if __name__ == "__main__":
    main()