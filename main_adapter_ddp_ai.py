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
from torch.utils.data import DataLoader, DistributedSampler
from nets.swinir import SwinIRAdapter, SwinIR
from utils.utils_dist import *
from utils.utils_data import print_config_as_table, load_config
from predict import evaluate_with_lrhr_pair, evaluate_with_hr
from predict_adapter import calculate_adapter_avg_psnr
import warnings
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import os
os.environ['MASTER_ADDR'] = '172.16.6.1'  # Replace '192.168.1.1' with the actual master node IP
os.environ['MASTER_PORT'] = '1234'        # Replace '12345' with the actual port if needed
# Filter out the specific warning
warnings.filterwarnings("ignore", message="Leaking Caffe2 thread-pool after fork. (function pthreadpool)")

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
parser.add_argument('--train_swinir', action='store_true', default=False, help='Train SwinIR model')
parser.add_argument('--dist', action='store_true', default=False, help='Enable distributed training')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

DEBUG = args.debug
train_swinir = args.train_swinir

print('Using train_swinir: {}'.format(train_swinir))

def process_config(config):
    config['train']['resume'] = config['train'].get('resume_optimizer') is not None and config['network'].get('resume_network') is not None

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
    return config

def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, config):
    setup(rank, world_size)

    if train_swinir:
        config['train']['seed'] = 2024
        config['train']['gpu_ids'] = [0,1,2,3]

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        print(f'Using GPU: {rank}')
    else:
        sys.exit('No GPU available')

    random.seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    torch.manual_seed(config['train']['seed'])
    torch.cuda.manual_seed_all(config['train']['seed'])

    if not DEBUG and rank == 0:
        if not os.path.exists(config['train']['save_path']):
            os.makedirs(config['train']['save_path'])

        with open(os.path.join(config['train']['save_path'], 'config.yaml'), 'w') as file:
            yaml.dump(config, file)

        wandb.init(
            project='SwinIR',
            id=config['train']['wandb_id'],
            name=config['train']['wandb_name'],
            config=config,
            resume='allow'
        )

    train_set = SuperResolutionYadaptDataset(config=config['train'])
    test_set = SuperResolutionYadaptDataset(config=config['test'])

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set,
                              batch_size=config['train']['batch_size'] // world_size,
                              shuffle=False,
                              num_workers=config['train']['num_workers'],
                              drop_last=True,
                              pin_memory=True,
                              sampler=train_sampler)

    test_loader = DataLoader(test_set,
                             batch_size=config['test']['batch_size'],
                             shuffle=False,
                             num_workers=config['test']['num_workers'],
                             drop_last=False,
                             pin_memory=True,
                             sampler=test_sampler)

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

    model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
    pretrained_model = torch.load(model_path, map_location='cuda:{}'.format(rank))
    param_key_g = 'params'
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=False)

    if config['network']['freeze_network']:
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.self_attention.parameters():
            param.requires_grad = True
        
        for param in model.adapt_conv.parameters():
            param.requires_grad = True

    if config['train']['loss'] == 'l1':
        criterion = torch.nn.L1Loss()

    device = torch.device('cuda:{}'.format(rank))
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if config['network']['resume_network']:
        checkpoint = torch.load(config['network']['resume_network'], map_location='cuda:{}'.format(rank))
        model.load_state_dict(checkpoint)
        print('Resume from checkpoint from {}'.format(config['network']['resume_network']))

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
            checkpoint = torch.load(config['train']['resume_optimizer'], map_location='cuda:{}'.format(rank))
            optimizer.load_state_dict(checkpoint)
            print('Resume from optimizer from {}'.format(config['train']['resume_optimizer']))
        else:
            optimizer = torch.optim.Adam(optim_params,
                                        lr=config['train']['lr'],
                                        betas=(0.9, 0.999),
                                        weight_decay=config['train']['weight_decay'])

    if config['train']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=config['train']['milestones'],
                                                        gamma=config['train']['gamma'])

    if config['network']['resume_network'] and config['train']['resume_optimizer']:
        step_network = config['network']['resume_network'].split('/')[-1].split('.')[0].split('_')[0]
        step_optimizer = config['train']['resume_optimizer'].split('/')[-1].split('.')[0].split('_')[0]
        assert step_network == step_optimizer, 'Network and Optimizer should have the same step.'
        current_step = int(step_network)
    else:
        current_step = 0

    if rank == 0:
        print_config_as_table(config, name=config['train']['wandb_name'])

    smooth_loss = []
    running_loss = 0.0
    for epoch in range(10000000000):
        train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            train_LR, train_HR, y_adapt = train_data
            train_HR = train_HR.cuda(rank, non_blocking=True)
            train_LR = train_LR.cuda(rank, non_blocking=True)
            y_adapt = y_adapt.cuda(rank, non_blocking=True)

            scheduler.step(current_step)

            if DEBUG and rank == 0:
                from PIL import Image

                HR_image = train_HR.cpu().permute(0,2,3,1).numpy()
                HR_image = HR_image[0]
                HR_image = HR_image * 255
                HR_image = HR_image.astype(np.uint8)
                HR_image = Image.fromarray(HR_image)
                HR_image.save('HR_image.png')

                LR_image = train_LR.cpu().permute(0,2,3,1).numpy()
                LR_image = LR_image[0]
                LR_image = LR_image * 255
                LR_image = LR_image.astype(np.uint8)
                LR_image = Image.fromarray(LR_image)
                LR_image.save('LR_image.png')

            optimizer.zero_grad()

            if train_swinir:
                output = model(train_LR)
            else:
                output = model(train_LR, y_adapt)

            if DEBUG and rank == 0:
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

            if current_step % config['train']['step_print'] == 0 and rank == 0:
                print('Epoch: {:d}, Step: {:d}, Loss: {:.4f}, Smoothed Loss: {:.4f}, LR: {:.8f}'.format(epoch, current_step, loss.item(), loss_smooth, scheduler.get_last_lr()[0]))
                wandb.log({"Epoch": epoch, "Step": current_step, "Loss": loss.item(), "Smoothed Loss": loss_smooth, "Learning Rate": scheduler.get_last_lr()[0]})

            if current_step % config['train']['step_save'] == 0 and rank == 0:
                torch.save(model.state_dict(), os.path.join(config['train']['save_path'], '{:d}_model.pth'.format(current_step)))
                torch.save(optimizer.state_dict(), os.path.join(config['train']['save_path'], '{:d}_optimizer.pth'.format(current_step)))
            
            if current_step % config['train']['step_test'] == 0 and rank == 0:
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
                        avg_psnr = calculate_adapter_avg_psnr(test_set, model, yadapt=True, scale=config['train']['scale'])
                    print('Epoch: {:d}, Step: {:d}, Avg PSNR: {:.4f}'.format(epoch, current_step, avg_psnr))
                    wandb.log({"Epoch": epoch, "Step": current_step, "Avg PSNR": avg_psnr})
                    model.train()

    cleanup()

def main():
    config_path = '/home/mayanze/PycharmProjects/SwinTF/config/Set5_ddp.yaml' 
    config = load_config(config_path)
    config = process_config(config)

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()