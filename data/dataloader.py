# This is the file to load the data from the dataset.

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

import sys
sys.path.append('/home/mayanze/PycharmProjects/SwinTF')


from data.data_utils import get_all_images
from data.extract_sam_features import extract_sam_model
# from extract_sam_features import extract_sam_model



def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))



# Write a super resolution dataset class

class SuperResolutionDataset(Dataset):
    def __init__(self, config):
        super(SuperResolutionDataset, self).__init__()
        self.config = config if isinstance(config, dict) else None
        self.mode = config['mode'] if self.config is not None else 'train'
        self.scale = config['scale'] if self.config is not None else 4
        self.patch_size = config['patch_size'] if self.config is not None else 96
        self.LR_size = self.patch_size // self.scale 
        
        if self.mode == 'train':
            self.HR_path = config['train_HR'] if self.config is not None else 'dataset/trainsets/trainH/DIV2K'
            self.LR_path = config['train_LR'] if self.config is not None else 'dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic'
            # Add in scale size in LR_path
            self.LR_path = os.path.join(self.LR_path, 'X{}'.format(self.scale))
        
        elif self.mode == 'test':
            self.HR_path = config['test_HR'] if self.config is not None else 'dataset/testsets/Set5'
            self.LR_path = config['test_LR'] if self.config is not None else 'dataset/testsets/Set5/LRbicx4'

        self.HR_images = get_all_images(self.HR_path)
        self.LR_images = get_all_images(self.LR_path)

        assert len(self.HR_images) == len(self.LR_images), 'HR and LR images should have the same length.'

    def __len__(self):
        return len(self.HR_images)
    
    def __getitem__(self, idx):
        HR_image = Image.open(self.HR_images[idx]) # 彩色图像读进来
        # Mode Crop 保证可以整除
        HR_image = modcrop(HR_image, self.scale)


        LR_image = Image.open(self.LR_images[idx])
        LR_image = np.array(LR_image) 

        if self.mode == 'train':

            H, W, C = LR_image.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.LR_size))
            rnd_w = random.randint(0, max(0, W - self.LR_size))
            LR_image = LR_image[rnd_h:rnd_h + self.LR_size, rnd_w:rnd_w + self.LR_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.scale), int(rnd_w * self.scale)
            HR_image = HR_image[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            mode = random.randint(0, 7)
            LR_image, HR_image = augment_img(LR_image, mode=mode), augment_img(HR_image, mode=mode)

        # To numpy
        HR_image = np.array(HR_image) / 255.0
        LR_image = np.array(LR_image) / 255.0

        # To tensor
        HR_image = torch.from_numpy(HR_image).permute(2, 0, 1).float()
        LR_image = torch.from_numpy(LR_image).permute(2, 0, 1).float()

        return LR_image, HR_image

class SuperResolutionYadaptDataset(Dataset):
    def __init__(self, config):
        super(SuperResolutionYadaptDataset, self).__init__()
        self.config = config if isinstance(config, dict) else None
        self.mode = config['mode'] if self.config is not None else 'train'
        self.scale = config['scale'] if self.config is not None else 4
        self.patch_size = config['patch_size'] if self.config is not None else 96
        self.LR_size = self.patch_size // self.scale 
        
        self.model = extract_sam_model(model_path=config['pretrained_sam'], image_size = config['pretrained_sam_img_size'])

        if self.mode == 'train':
            self.HR_path = config['train_HR'] if self.config is not None else 'dataset/trainsets/trainH/DIV2K'
            self.LR_path = config['train_LR'] if self.config is not None else 'dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic'
            # Add in scale size in LR_path
            self.LR_path = os.path.join(self.LR_path, 'X{}'.format(self.scale))
            # self.yadapt_features = np.load(config['train_yadapt'])
        
        elif self.mode == 'test':
            self.HR_path = config['test_HR'] if self.config is not None else 'dataset/testsets/Set5'
            self.LR_path = config['test_LR'] if self.config is not None else 'dataset/testsets/Set5/LRbicx4'
            # self.yadapt_features = np.load(config['test_yadapt'])

        self.HR_images = get_all_images(self.HR_path)
        self.LR_images = get_all_images(self.LR_path)

        assert len(self.HR_images) == len(self.LR_images), 'HR and LR images should have the same length.'

    def __len__(self):
        return len(self.HR_images)
    
    def __getitem__(self, idx):
        HR_image = Image.open(self.HR_images[idx]) # 彩色图像读进来
        # Mode Crop 保证可以整除
        HR_image = modcrop(HR_image, self.scale)


        LR_image = Image.open(self.LR_images[idx])
        LR_image = np.array(LR_image) 

        if self.mode == 'train':

            H, W, C = LR_image.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.LR_size))
            rnd_w = random.randint(0, max(0, W - self.LR_size))
            LR_image = LR_image[rnd_h:rnd_h + self.LR_size, rnd_w:rnd_w + self.LR_size, :]


            # XXX Future Fix Move out of the function
            # LR_image to Batch LR image
            batch_img = torch.from_numpy(LR_image).permute(2, 0, 1).unsqueeze(0).float()
            _, y1, y2, y3 = self.model(batch_img)
            # Concatenate the features
            yadapt_features = np.concatenate((y1.squeeze(0).detach().cpu().numpy(), y2.squeeze(0).detach().cpu().numpy(), y3.squeeze(0).detach().cpu().numpy()), axis=0)
            # Print the size of yadapt_features
            # print(yadapt_features.shape) # 3480x3x3 
            yadapt_features = torch.from_numpy(yadapt_features).float()
            
 
            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.scale), int(rnd_w * self.scale)
            HR_image = HR_image[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            mode = random.randint(0, 7)
            LR_image, HR_image = augment_img(LR_image, mode=mode), augment_img(HR_image, mode=mode)

            # To numpy
            HR_image = np.array(HR_image) / 255.0
            LR_image = np.array(LR_image) / 255.0
    
            # To tensor
            HR_image = torch.from_numpy(HR_image).permute(2, 0, 1).float()
            LR_image = torch.from_numpy(LR_image).permute(2, 0, 1).float()
    
            return LR_image, HR_image, yadapt_features

        if self.mode == "test":
            # LR_image to Batch LR image
            # TODO 这里修改一下 LR HR 的mod方式，应该是 LR 先算，然后对应到 HR
            # HR = [:LR_imageshape0*self.scale, :LR_imageshape1*self.scale]
            # 首先对 LR 图像进行 mod_crop 必须能被 48 整除
            LR_image = modcrop(LR_image, 48) 
            # 然后对应的 HR 图像也要变换 
            HR_image = modcrop(HR_image, 48) / 255.0




            # 设置一个 assert 保证 HR 和 LR 放大后的大小是一样的
            assert HR_image.shape[0] == LR_image.shape[0] * self.scale and HR_image.shape[1] == LR_image.shape[1] * self.scale, "HR and LR should have the same size after modcrop"

            # 参考下面的精神，将 LR_image 转换成 batch 形式
            # LR_image 的形状 [48*x, 48*y, 3] -> [x, y, 3, 48, 48] -> [x*y, 3, 48, 48]
            x, y, _ = LR_image.shape
            batch_LR_image = LR_image.reshape(x//48, 48, y//48, 48, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 48, 48, 3).transpose(0, 3, 1, 2)
            # 然后将 batch_LR_image 转换成 tensor
            batch_LR_image_sam = torch.from_numpy(batch_LR_image).float()
            # 然后将 batch_LR_image 输入到模型中
            _, y1, y2, y3 = self.model(batch_LR_image_sam)
            
            # import matplotlib.pyplot as plt
            # plt.imshow(batch_LR_image[0,0,:,:])
            # plt.savefig('test.png')
            
            
            
            # Concatenate the features
            yadapt_features = np.concatenate((y1.squeeze(0).detach().cpu().numpy(), y2.squeeze(0).detach().cpu().numpy(), y3.squeeze(0).detach().cpu().numpy()), axis=1)
            # Print the size of yadapt_features
            # print(yadapt_features.shape) # 3480x3x3
            batch_yadapt_features = torch.from_numpy(yadapt_features).float()
            # 这里由于 vit 会把 B 和 C 合成一个维度，所以这里要把 batch_yadapt_features 的维度转换一下，变成 [xy/3, 1280*3, 3, 3]
            # batch_yadapt_features = batch_yadapt_features.reshape(yadapt_features.shape[0]//3, 1280*3, 3, 3)
            # assert 到这里 batch_yadapt_features 和 batch_LR_image 的 batch_size 是一样的
            assert batch_yadapt_features.shape[0] == batch_LR_image.shape[0], "batch_yadapt_features and batch_LR_image should have the same batch_size"

            batch_LR_image = batch_LR_image / 255.0
            batch_LR_image = torch.from_numpy(batch_LR_image).float()

            return batch_LR_image, HR_image, batch_yadapt_features,(x,y)






if __name__ == "__main__":
    # dataset class 要有 __init__ 和 __len__ 和 __getitem__ 三个函数
    # __init__ 中的参数：config（最终），最终是需要 config 来整体设计，一个 config 走天下
    import yaml
    config_path = '/home/mayanze/PycharmProjects/SwinTF/config/example copy.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    # DIV2K = SuperResolutionDataset(config=config['train'])
    # LR_image, HR_image = DIV2K.__getitem__(0)
    # print(LR_image.shape, HR_image.shape)

    # test_set = SuperResolutionDataset(config=config['test'])
    # LR_image, HR_image = test_set.__getitem__(0)
    # print(LR_image.shape, HR_image.shape)

    # DIV2K = SuperResolutionYadaptDataset(config=config['train'])
    # LR_image, HR_image, yadapt = DIV2K.__getitem__(0)
    # print(LR_image.shape, HR_image.shape, yadapt.shape)

    # test_set = SuperResolutionYadaptDataset(config=config['test'])
    # LR_image, HR_image, yadapt, (x,y)= test_set.__getitem__(0)
    # print(LR_image.shape, HR_image.shape, yadapt.shape)

    # 对yadapt_features进行测试, 对同一个位置跑两次，结果应该是一样的
    test_set = SuperResolutionYadaptDataset(config=config['test'])
    r0_LR_image, r0_HR_image, r0_yadapt, (x,y)= test_set.__getitem__(0)
    test_set2 = SuperResolutionYadaptDataset(config=config['test'])
    r1_LR_image, r1_HR_image, r1_yadapt, (x,y)= test_set2.__getitem__(0)

    # 检查是否完全一样
    print(np.allclose(r0_LR_image, r1_LR_image))
    print(np.allclose(r0_HR_image, r1_HR_image))
    print(np.allclose(r0_yadapt, r1_yadapt))

    #=========================================================================
    ### 检查出来确实是不一样，yadapt_features 为什么不一样，继续检查
    #=========================================================================