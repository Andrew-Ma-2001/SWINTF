import os 
import yaml
import math
import torch 
import time
import wandb
import random
import numpy as np
from data.dataloader import SuperResolutionYadaptDataset
from torch.utils.data import DataLoader
from nets.swinir import SwinIRAdapter
from utils.utils_image import calculate_psnr, permute_squeeze

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

# TODO Matlab 的降采样这里用
# TODO 用 Urban100 的数据集来测试 - basic SR 应该有

# =================================================
# 0 Config，Global Parameters 部分
# =================================================
config_path = 'config/example copy.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

print(config)

gpu_ids = config['train']['gpu_ids']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

if torch.cuda.is_available():
    print('Using GPU: {}'.format(gpu_ids))

seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

wandb.init(project='SwinIR', config=config)
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

# 3.2 设计断点续训的机制
if config['network']['resume_network']:
    checkpoint = torch.load(config['network']['resume_network'])
    model.load_state_dict(checkpoint['state_dict'])
    print('Resume from checkpoint from {}'.format(config['network']['resume_network']))

# ================================================
# 4 Loss, Optimizer 和 Scheduler 部分
# ================================================

# 4.1 Loss 定义
if config['train']['loss'] == 'l1':
    criterion = torch.nn.L1Loss()

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
                                    weight_decay=config['train']['weight_decay'])
        checkpoint = torch.load(config['train']['resume_optimizer'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Resume from optimizer from {}'.format(config['train']['resume_optimizer']))
    else:
        optimizer = torch.optim.Adam(optim_params,
                                    lr=config['train']['lr'],
                                    weight_decay=config['train']['weight_decay'])

# 4.3 Scheduler 定义
if config['train']['scheduler'] == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=config['train']['milestones'], # milestones: [250000, 400000, 450000, 475000, 500000]
                                                    gamma=config['train']['gamma']) # gamma: 0.5


# ==================================================
# 5 Training 部分
# ==================================================
# 5.1 训练初始化，找到 Step
device = torch.device('cuda' if gpu_ids is not None else 'cpu')

model.train()
model.cuda()
model = torch.nn.DataParallel(model) # 4，5 visable 变成 0，1


date_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
# 保存现在的时间，创建一个文件夹
config['train']['save_path'] = os.path.join(config['train']['save_path'], config['train']['type'] + '_' + date_time)

if not os.path.exists(config['train']['save_path']):
    os.makedirs(config['train']['save_path'])

# 保存现在用的 config
with open(os.path.join(config['train']['save_path'], 'config.yaml'), 'w') as file:
    yaml.dump(config, file)



# 用 Step 来记录训练的次数
if config['network']['resume_network'] and config['train']['resume_optimizer']:
    # 步长保存为模型的名字 {step}_{model_name}.pth
    step_network = config['train']['resume_network'].split('/')[-1].split('.')[0].split('_')[0]
    # 步长保存为优化器的名字 {step}_{optimizer_name}.pth
    step_optimizer = config['train']['resume_optimizer'].split('/')[-1].split('.')[0].split('_')[0]
    assert step_network == step_optimizer, 'Network and Optimizer should have the same step.'
    step = int(step_network)

else:
    step = 0

# 5.2 更新 Scheduler
if step > 0:
    for _ in range(step):
        scheduler.step()

# 5.3 开始训练
for epoch in range(10000000000):
    for _, train_data in enumerate(train_loader):
        # 5.3.1 数据准备
        train_LR, train_HR, y_adapt = train_data
        train_HR = train_HR.cuda()
        train_LR = train_LR.cuda()
        y_adapt = y_adapt.cuda()

        # 5.3.2 训练模型
        optimizer.zero_grad()
        output = model.forward(train_LR, y_adapt)
        loss = criterion(output, train_HR)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        # 5.3.3 打印训练信息
        if step % config['train']['step_print'] == 0:
            print('Epoch: {:d}, Step: {:d}, Loss: {:.4f}, LR: {:.8f}'.format(epoch, step, loss.item(), scheduler.get_last_lr()[0]))
            wandb.log({"Epoch": epoch, "Step": step, "Loss": loss.item(), "Learning Rate": scheduler.get_last_lr()[0]})

        # 5.3.4 保存模型
        if step % config['train']['step_save'] == 0:
            torch.save(model.state_dict(), os.path.join(config['train']['save_path'], '{:d}_model.pth'.format(step)))
            torch.save(optimizer.state_dict(), os.path.join(config['train']['save_path'], '{:d}_optimizer.pth'.format(step)))
        
        # 5.3.5 测试模型
        # if step % config['train']['step_test'] == 0:
        #     total_psnr = 0
        #     total_loss = 0
        #     model.eval()
        #     with torch.no_grad():
        #         for _, test_data in enumerate(test_loader):
        #             test_LR, test_HR, y_adapt_feature = test_data
        #             test_HR = test_HR.cuda(device=device)
        #             test_LR = test_LR.cuda(device=device)
        #             y_adapt_feature = y_adapt_feature.cuda(device=device)

        #             output = model.forward(test_LR, y_adapt_feature)
        #             loss = criterion(output, test_HR)

        #             total_loss += loss.item()

        #             # 改变维度，计算 PSNR
        #             output = permute_squeeze(output)
        #             test_HR = permute_squeeze(test_HR)

        #             current_psnr = calculate_psnr(output, test_HR, border=config['test']['scale'], max_val=1)
        #             total_psnr += current_psnr
        #         avg_psnr = total_psnr / len(test_loader)
        #         avg_loss = total_loss / len(test_loader)
        #         print('Epoch: {:d}, Step: {:d}, Avg PSNR: {:.4f}, Avg Loss: {:.4f}'.format(epoch, step, avg_psnr, avg_loss))
        #         wandb.log({"Epoch": epoch, "Step": step, "Avg PSNR": avg_psnr, "Avg Loss": avg_loss})
        #     model.train()
