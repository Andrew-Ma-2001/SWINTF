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
    print(yadapt_feature.shape)


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


def check_test_consistency(config):
    from data.dataloader import SuperResolutionYadaptDataset
    import numpy as np
    # 对yadapt_features进行测试, 对同一个位置跑两次，结果应该是一样的
    test_set = SuperResolutionYadaptDataset(config=config['test'])
    r0_LR_image, r0_HR_image, r0_yadapt, _, _, _ = test_set.__getitem__(0)
    test_set2 = SuperResolutionYadaptDataset(config=config['test'])
    r1_LR_image, r1_HR_image, r1_yadapt,  _, _, _ = test_set2.__getitem__(0)

    # 检查是否完全一样
    print(np.allclose(r0_LR_image, r1_LR_image))
    print(np.allclose(r0_HR_image, r1_HR_image))
    print(np.allclose(r0_yadapt, r1_yadapt))


def check_yadapt_files():
    import numpy as np
    file1 = '/home/mayanze/PycharmProjects/SwinTF/dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic/X0_yadapt/0001x2_yadapt.npy'
    file2 = '/home/mayanze/PycharmProjects/SwinTF/dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic/X1_yadapt/0001x2_yadapt.npy'
    yadapt1 = np.load(file1)
    yadapt2 = np.load(file2)
    print(np.allclose(yadapt1, yadapt2))



def check_two_npy_file():
    # Load in two npy files and check if is the same after sorting
    import numpy as np

    file_path1 = '/home/mayanze/PycharmProjects/SwinTF/max_yadapt_values.npy'
    file_path2 = '/home/mayanze/PycharmProjects/SwinTF/max_yadapt_values2.npy'

    yadapt_feature1 = np.load(file_path1)
    yadapt_feature2 = np.load(file_path2)

    yadapt_feature1 = np.sort(yadapt_feature1)
    yadapt_feature2 = np.sort(yadapt_feature2)

    print(np.allclose(yadapt_feature1, yadapt_feature2))

# 把预处理函数写在外面，不能写在里面了
def check_precompute():
    import numpy as np
    import sys
    sys.path.append('/home/mayanze/PycharmProjects/SwinTF/')
    from PIL import Image
    from utils.utils_data import get_all_images, process_batch
    from nets.build_sam import extract_sam_model
    from utils.utils_image import augment_img

    LR_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic/X1'
    LR_size = 48
    pretrained_sam_img_size = 48
    use_cuda = True
    save_path = LR_path + '_yadapt_aug'
    model = extract_sam_model(model_path='/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth', image_size = 1024)
    # only use 0,1 gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'



    if use_cuda:
        model = model.cuda()
        model.image_encoder = torch.nn.DataParallel(model.image_encoder)

    LR_images = get_all_images(LR_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert len(os.listdir(save_path)) == 0, "The save_path should be empty"

    modes = [0,1,2,3,4,5,6,7] # 8 modes

    for mode in modes:
        for idx in range(len(LR_images)):
            LR_image = Image.open(LR_images[idx])
            LR_image = np.array(LR_image)
            patches = []

            # Cut the LR_image into patches
            for y in range(0, LR_image.shape[0] - LR_size + 1, LR_size):
                for x in range(0, LR_image.shape[1] - LR_size + 1, LR_size):
                    patch = LR_image[y:y + LR_size, x:x + LR_size,:]
                    patch = augment_img(patch, mode)
                    patches.append((patch, x, y))

            batch_LR_image = np.zeros((len(patches), 3, pretrained_sam_img_size, pretrained_sam_img_size))
            for i, (patch, _, _) in enumerate(patches):
                batch_LR_image[i] = patch.transpose(2, 0, 1)

            # 这里要把 48x48 变成 1024x1024 建一个更大的矩阵
            large_img = np.zeros((batch_LR_image.shape[0], 3, 1024, 1024))
            large_img[:, :, :48, :48] = batch_LR_image
            # 然后将 batch_LR_image 转换成 tensor
            batch_LR_image_sam = torch.from_numpy(large_img).float()
            # 然后将 batch_LR_image 输入到模型中
            inferece_batch_size = 25

            if batch_LR_image_sam.shape[0] <= inferece_batch_size:
                if use_cuda:
                    batch_LR_image_sam = batch_LR_image_sam.cuda()

                with torch.no_grad():
                    _, y1, y2, y3 = model.image_encoder(batch_LR_image_sam)
                    y1, y2, y3 = y1.cpu().numpy(), y2.cpu().numpy(), y3.cpu().numpy()
            else:
                if use_cuda:
                    y1, y2, y3 = process_batch(batch_LR_image_sam, model.image_encoder, inferece_batch_size)

            y1, y2, y3 = y1[:, :, :3, :3], y2[:, :, :3, :3], y3[:, :, :3, :3]
            yadapt_features = np.concatenate((y1, y2, y3), axis=1)

            # 对于 yadapt_features 分别保存
            for i in range(yadapt_features.shape[0]):
                save_name = os.path.join(save_path, os.path.basename(LR_images[idx]).split(".")[0]+'_'+str(i)+'_'+str(mode)+'_yadapt.npy')
                np.save(save_name, yadapt_features[i])
                print('Save {}'.format(save_name))


def check_plot_yadapt_distribution():
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    file_dir = "/home/mayanze/PycharmProjects/SwinTF/dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic/X2_yadapt"
    files = os.listdir(file_dir)
    files = sorted(files)
    files = [os.path.join(file_dir, file) for file in files if file.endswith(".npy")]
    max_values = []
    i = 0
    for num in tqdm(range(files.__len__())):
        data = np.load(files[num])
        # Make the x axis be -50 to 50
        plt.hist(data.flatten(), bins=100)
        # Save the figure
        plt.savefig(f"yadapt_distribution_{i}.png")
        # Reset the plot
        plt.clf()
        i += 1
        max_values.append(np.max(data))

    print(max(max_values))
    print(min(max_values))

    print(max_values)

import sys
sys.path.append('/home/mayanze/PycharmProjects/SwinTF/')
check_precompute()

