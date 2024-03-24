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


import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore")

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


# =================================================
# 0 Config，Global Parameters 部分
# =================================================
config_path = '/home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml'
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

stride = test_set.pretrained_sam_img_size - test_set.overlap
scale_factor = test_set.scale
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


# 加载预训练 SwinIR 模型
model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
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
        optimizer.load_state_dict(checkpoint)
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
model = torch.nn.DataParallel(model)

# 3.2 设计断点续训的机制
if config['network']['resume_network']:
    checkpoint = torch.load(config['network']['resume_network'])
    model.load_state_dict(checkpoint)
    print('Resume from checkpoint from {}'.format(config['network']['resume_network']))

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
    step_network = config['network']['resume_network'].split('/')[-1].split('.')[0].split('_')[0]
    # 步长保存为优化器的名字 {step}_{optimizer_name}.pth
    step_optimizer = config['train']['resume_optimizer'].split('/')[-1].split('.')[0].split('_')[0]
    # assert step_network == step_optimizer, 'Network and Optimizer should have the same step.'
    step = int(step_network)

else:
    step = 0

# 5.2 更新 Scheduler
if step > 0:
    for _ in range(step):
        scheduler.step()

# 5.3 开始训练
smooth_loss = []
running_loss = 0.0
for epoch in range(10000000000):
    for _, train_data in enumerate(train_loader):
        # 5.3.1 数据准备
        train_LR, train_HR, y_adapt = train_data
        train_HR = train_HR.cuda()
        train_LR = train_LR.cuda()

        # Make all the y_adapt be zero
        # y_adapt = torch.zeros_like(y_adapt)

        y_adapt = y_adapt.cuda()
        # 5.3.2 训练模型
        optimizer.zero_grad()
        output = model.forward(train_LR, y_adapt)
        loss = criterion(output, train_HR)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        running_loss += loss.item()
        smooth_loss.append(loss.item())
        if len(smooth_loss) > 100:
            smooth_loss.pop(0)
        loss_smooth = sum(smooth_loss) / len(smooth_loss)

        # 5.3.3 打印训练信息
        if step % config['train']['step_print'] == 0:
            print('Epoch: {:d}, Step: {:d}, Loss: {:.4f}, Smoothed Loss: {:.4f}, LR: {:.8f}'.format(epoch, step, loss.item(), loss_smooth, scheduler.get_last_lr()[0]))
            wandb.log({"Epoch": epoch, "Step": step, "Loss": loss.item(), "Smoothed Loss": loss_smooth, "Learning Rate": scheduler.get_last_lr()[0]})

        # 5.3.4 保存模型
        if step % config['train']['step_save'] == 0:
            torch.save(model.state_dict(), os.path.join(config['train']['save_path'], '{:d}_model.pth'.format(step)))
            torch.save(optimizer.state_dict(), os.path.join(config['train']['save_path'], '{:d}_optimizer.pth'.format(step)))
        
        # 5.3.5 测试模型
        # if step % config['train']['step_test'] == 0:
        #     total_psnr = 0
        #     total_loss = 0
        #     model.eval()

        #     for iter, test_data in enumerate(test_set):
        #         batch_LR_image, HR_image, batch_yadapt_features = test_data[0], test_data[1], test_data[2]
        #         patches, (img_height, img_width), (padded_height, padded_width) = test_data[3], test_data[4], test_data[5]
        #         test_HR = HR_image.astype(np.float32) / 255.0
        #         test_HR = torch.from_numpy(test_HR).permute(2, 0, 1).float()

        #         test_HR = test_HR.cuda()
        #         batch_LR_image = batch_LR_image.cuda()
        #         batch_yadapt_features = batch_yadapt_features.cuda()
        #         with torch.no_grad():
        #             batch_Pre_image = model.forward(batch_LR_image, batch_yadapt_features)
        #             batch_Pre_image = batch_Pre_image.clamp(0, 1).cpu().permute(0,2,3,1).numpy()
                    
        #             batch_Pre_image = batch_Pre_image.astype(np.uint8)
        #             super_res_image = np.zeros((stride * (padded_height // stride) * scale_factor, stride * (padded_width // stride) * scale_factor, 3), dtype=np.uint8)

        #             for (patch, x, y), pre_image in zip(patches, batch_Pre_image):
        #                 num_y = y // stride
        #                 num_x = x // stride
        #                 patch = pre_image[(test_set.overlap//2)*scale_factor:(test_set.pretrained_sam_img_size-test_set.overlap//2)*scale_factor, (test_set.overlap//2)*scale_factor:(test_set.pretrained_sam_img_size-test_set.overlap//2)*scale_factor, :]
        #                 super_res_image[num_y*stride*scale_factor:(num_y+1)*stride*scale_factor, num_x*stride*scale_factor:(num_x+1)*stride*scale_factor,:] = patch

        #             super_res_image = super_res_image[:img_height*scale_factor-test_set.overlap*scale_factor, :img_width*scale_factor - test_set.overlap*scale_factor]
        #             loss = criterion(super_res_image, test_HR)
        #             total_loss += loss.item()

        #             total_psnr += calculate_psnr(output, test_HR, border=config['test']['scale'], max_val=1)
        #         avg_psnr = total_psnr / len(test_loader)
        #         avg_loss = total_loss / len(test_loader)
        #         print('Epoch: {:d}, Step: {:d}, Avg PSNR: {:.4f}, Avg Loss: {:.4f}'.format(epoch, step, avg_psnr, avg_loss))
        #         wandb.log({"Epoch": epoch, "Step": step, "Avg PSNR": avg_psnr, "Avg Loss": avg_loss})
        #     model.train()
        
