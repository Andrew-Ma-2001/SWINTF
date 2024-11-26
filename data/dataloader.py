# This is the file to load the data from the dataset.
import math
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import cv2
import sys
sys.path.append('/home/mayanze/PycharmProjects/SwinTF')
from compute_dataset import ImagePreprocessor


from data.extract_sam_features import extract_sam_model
from utils.utils_data import get_all_images, extract_patches, process_batch
from utils.utils_image import imresize_np, modcrop, augment_img

# Write a super resolution dataset class

DEBUG = False

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
        
        self.model = extract_sam_model(model_path=config['pretrained_sam'], image_size = 1024)
        self.preprocessor = ImagePreprocessor()

        if not DEBUG:
            self.use_cuda = config['yadapt_use_cuda']
        else:
            self.use_cuda = False

        if self.use_cuda:
            # Move the model to a specific 
            self.model = self.model.cuda()
            self.model.image_encoder = torch.nn.DataParallel(self.model.image_encoder)

        if self.mode == 'train':
            self.HR_path = config['train_HR'] # 'dataset/trainsets/trainH/DIV2K'
            self.LR_path = config['train_LR'] # 'dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic'
            # Add in scale size in LR_path
            self.LR_path = os.path.join(self.LR_path, 'X{}'.format(self.scale))
            # self.yadapt_features = np.load(config['train_yadapt'])
        
        elif self.mode == 'test':
            self.HR_path = config['test_HR'] # 'dataset/testsets/Set5'
            self.LR_path = config['test_LR'] # 'dataset/testsets/Set5/LRbicx4'

            # Check if path exists
            assert os.path.exists(self.HR_path), "HR_path should exist"
            assert os.path.exists(self.LR_path) or self.LR_path in ['BIC'], "LR_path should exist or be in ['BIC']"

        elif self.mode == 'pred':
            self.HR_path = config['pred_HR'] # TODO 这个参数要改，顺便把下面 self.HR_images = get_all_images(self.HR_path) 找个地方放
            self.LR_path = config['pred_LR']

            assert os.path.exists(self.LR_path), "LR_path should exist"

        # TODO 把这个参数写在外面
        self.overlap = 0
        self.pixel_mean = np.array([123.675, 116.28, 103.53])
        self.pixel_std = np.array([58.395, 57.12, 57.375])

        self.pretrained_sam_img_size = config['pretrained_sam_img_size']

        self.HR_images = get_all_images(self.HR_path)

        if self.LR_path == 'BIC':
            self.LR_images = self.HR_images
        else:
            self.LR_images = get_all_images(self.LR_path)

        assert len(self.HR_images) == len(self.LR_images), 'HR and LR images should have the same length.'

        #DEBUG 保存路径要写在 init 里面，或者在config里面传进来
        # Check if the yadapt_features exists
        # if self.mode == 'test':
        #     self.check_test_precompute()
        #     if self.config['precomputed']:
        #         self.check_test_precompute()
        #         if not self.precompute:
        #             self.precompute_test2() 
        
        # if self.mode == 'train':
        #     self.check_train_precompute()
        #     if self.config['precomputed']:
        #         self.check_train_precompute()
        #         if not self.precompute:
        #             self.precompute_train()

    
    def check_test_precompute(self):
        # TODO 还是要考虑算一半的情况
        if self.LR_path == 'BIC':
            save_path = self.HR_path + '_yadapt_aug'
        else:
            save_path = self.LR_path + '_yadapt_aug'
        
        if os.path.exists(save_path):
            assert len(os.listdir(save_path)) == len(self.HR_images), "The save_path stored files should have the same length as HR_images"
            self.precompute = True
        else:
            self.precompute = False

    def check_train_precompute(self):
        if self.LR_path == 'BIC':
            save_path = self.HR_path + '_yadapt'
        else:
            save_path = self.LR_path + '_yadapt'
        
        if os.path.exists(save_path):
            assert len(os.listdir(save_path)) == len(self.HR_images), "The save_path stored files should have the same length as HR_images"
            self.precompute = True
        else:
            self.precompute = False

    def precompute_test(self):
        if self.LR_path == 'BIC':
            save_path = self.HR_path + '_yadapt'
        else:
            save_path = self.LR_path + '_yadapt'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        assert len(os.listdir(save_path)) == 0, "The save_path should be empty"

        for i in range(len(self.HR_images)):
            LR_image, HR_image, yadapt, _ , _, _ = self.__getitem__(i)
            print(LR_image.shape, HR_image.shape, yadapt.shape)
            yadapt = yadapt.cpu().numpy()
            save_name = os.path.join(save_path, os.path.basename(self.LR_images[i]).split(".")[0]+'_yadapt.npy')
            np.save(save_name, yadapt)
            print('Save {}'.format(save_name))
        
        self.precompute = True
    
    def force_crop(self, image, size):
        # Crop the image to the size
        h, w = size
        img = image[:h, :w, :]
        return img

    def precompute_test2(self):
        if self.LR_path == 'BIC':
            save_path = self.HR_path + '_yadapt_aug'
        else:
            save_path = self.LR_path + '_yadapt_aug'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        assert len(os.listdir(save_path)) == 0, "The save_path should be empty"

        for idx in range(len(self.HR_images)):
            HR_image = Image.open(self.HR_images[idx]) # 彩色图像读进来
            HR_image = modcrop(HR_image, self.scale)
            if self.LR_path == 'BIC':
                LR_image = Image.open(self.HR_images[idx])
                # Ensure the image is read in three channels (RGB)
                LR_image = LR_image.convert('RGB')
                LR_image = modcrop(LR_image, self.scale)
                LR_image = imresize_np(LR_image/255.0, 1/self.scale)
                LR_image = np.array(LR_image*255.0)
            else:
                LR_image = Image.open(self.LR_images[idx])
                LR_image = LR_image.convert('RGB')
                LR_image = np.array(LR_image)
            LR_image = (LR_image - self.pixel_mean) / self.pixel_std

            try:
                assert HR_image.shape[0] == LR_image.shape[0] * self.scale and HR_image.shape[1] == LR_image.shape[1] * self.scale, "HR and LR should have the same size after modcrop"
            except:
                # HR_image = self.force_crop(HR_image, (LR_image.shape[0] * self.scale, LR_image.shape[1] * self.scale))
                LR_image = self.force_crop(LR_image, (HR_image.shape[0] // self.scale, HR_image.shape[1] // self.scale))
                print('HR and LR should have the same size after modcrop')
                assert HR_image.shape[0] == LR_image.shape[0] * self.scale and HR_image.shape[1] == LR_image.shape[1] * self.scale, "HR and LR should have the same size after modcrop"


            HR_image = HR_image[self.overlap:HR_image.shape[0]-self.overlap, self.overlap:HR_image.shape[1] - self.overlap]

            # super_res_image = super_res_image[:img_height*2-overlap*2, :img_width*2 - overlap*2]

            patches, (padded_height, padded_width), (img_height, img_width) = extract_patches(LR_image, self.pretrained_sam_img_size, self.overlap, constant_values=0.5)
            # 这里思考 precompute 的问题
            # 现在把 patches 以 batch 的形式先拼接
            batch_LR_image = np.zeros((len(patches), 3, self.pretrained_sam_img_size, self.pretrained_sam_img_size), dtype=np.float32)
            for i, (patch, _, _) in enumerate(patches):
                batch_LR_image[i] = patch.transpose(2, 0, 1)
            
            # 这里要把 48x48 变成 1024x1024 建一个更大的矩阵
            large_img = np.zeros((batch_LR_image.shape[0], 3, 1024, 1024))
            large_img[:, :, :48, :48] = batch_LR_image

            # 然后将 batch_LR_image 转换成 tensor
            batch_LR_image_sam = torch.from_numpy(large_img).float()
            # 然后将 batch_LR_image 输入到模型中

            if batch_LR_image_sam.shape[0] <= 15:
                if self.use_cuda:
                    batch_LR_image_sam = batch_LR_image_sam.cuda()
                with torch.no_grad():
                    _, y1, y2, y3 = self.model.image_encoder(batch_LR_image_sam)
                    y1, y2, y3 = y1.cpu().numpy(), y2.cpu().numpy(), y3.cpu().numpy()
            else:
                if self.use_cuda:
                    y1, y2, y3 = process_batch(batch_LR_image_sam, self.model.image_encoder, 15)
                # import matplotlib.pyplot as plt
                # plt.imshow(batch_LR_image[0,0,:,:])
                # plt.savefig('test.png')
                # Concatenate the features
            y1, y2, y3 = y1[:, :, :3, :3], y2[:, :, :3, :3], y3[:, :, :3, :3]
            yadapt_features = np.concatenate((y1, y2, y3), axis=1)

            batch_yadapt_features = torch.from_numpy(yadapt_features).float()
            # 这里由于 vit 会把 B 和 C 合成一个维度，所以这里要把 batch_yadapt_features 的维度转换一下，变成 [xy/3, 1280*3, 3, 3]
            # batch_yadapt_features = batch_yadapt_features.reshape(yadapt_features.shape[0]//3, 1280*3, 3, 3)
            # assert 到这里 batch_yadapt_features 和 batch_LR_image 的 batch_size 是一样的
            assert batch_yadapt_features.shape[0] == batch_LR_image.shape[0], "batch_yadapt_features and batch_LR_image should have the same batch_size"


            save_name = os.path.join(save_path, os.path.basename(self.LR_images[idx]).split(".")[0]+'_yadapt.npy')
            np.save(save_name, yadapt_features)
            print('Save {}'.format(save_name))

    def precompute_train(self):

        print("Warning This Function Has Not Consider Mode Augmentation")
        print("Check compute_dataset.py to see how to add mode augmentation")
        raise NotImplementedError

        if self.LR_path == 'BIC':
            save_path = self.HR_path + '_yadapt'
        else:
            save_path = self.LR_path + '_yadapt'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        assert len(os.listdir(save_path)) == 0, "The save_path should be empty"

        for idx in range(len(self.HR_images)):
            LR_image = Image.open(self.LR_images[idx])
            LR_image = np.array(LR_image)
            LR_image = (LR_image - self.pixel_mean) / self.pixel_std


            patches = [] 
            # Cut the LR_image into patches
            for y in range(0, LR_image.shape[0] - self.LR_size + 1, self.LR_size):
                for x in range(0, LR_image.shape[1] - self.LR_size + 1, self.LR_size):
                    patch = LR_image[y:y + self.LR_size, x:x + self.LR_size,:]
                    patches.append((patch, x, y))

            batch_LR_image = np.zeros((len(patches), 3, self.pretrained_sam_img_size, self.pretrained_sam_img_size))
            for i, (patch, _, _) in enumerate(patches):
                batch_LR_image[i] = patch.transpose(2, 0, 1)

            
            # 这里要把 48x48 变成 1024x1024 建一个更大的矩阵
            large_img = np.zeros((batch_LR_image.shape[0], 3, 1024, 1024))
            large_img[:, :, :48, :48] = batch_LR_image
            # 然后将 batch_LR_image 转换成 tensor
            batch_LR_image_sam = torch.from_numpy(large_img).float()
            # 然后将 batch_LR_image 输入到模型中
            inferece_batch_size = 5
            
            if batch_LR_image_sam.shape[0] <= inferece_batch_size:
                if self.use_cuda:
                    batch_LR_image_sam = batch_LR_image_sam.cuda()
                with torch.no_grad():
                    _, y1, y2, y3 = self.model.image_encoder(batch_LR_image_sam)
                    y1, y2, y3 = y1.cpu().numpy(), y2.cpu().numpy(), y3.cpu().numpy()
            else:
                if self.use_cuda:
                    y1, y2, y3 = process_batch(batch_LR_image_sam, self.model.image_encoder, inferece_batch_size)
            # import matplotlib.pyplot as plt
            # plt.imshow(batch_LR_image[0,0,:,:])
            # plt.savefig('test.png')
            # Concatenate the features
            y1, y2, y3 = y1[:, :, :3, :3], y2[:, :, :3, :3], y3[:, :, :3, :3]
            yadapt_features = np.concatenate((y1, y2, y3), axis=1)

            save_name = os.path.join(save_path, os.path.basename(self.LR_images[idx]).split(".")[0]+'_yadapt.npy')
            np.save(save_name, yadapt_features)
            print('Save {}'.format(save_name))
            

    def __len__(self):
        return len(self.HR_images)
    
    def get_hr_lr_pair(self, idx):
        HR_image = Image.open(self.HR_images[idx]) # 彩色图像读进来
        # Mode Crop 保证可以整除
        HR_image = modcrop(HR_image, self.scale)
        LR_image = Image.open(self.LR_images[idx])
        LR_image = LR_image.convert('RGB')
        LR_image = np.array(LR_image) 

        if self.LR_path == 'BIC':
            LR_image = modcrop(LR_image, self.scale)
            LR_image = imresize_np(LR_image/255.0, 1/self.scale)
            LR_image = np.array(LR_image*255.0)

        return HR_image, LR_image

    def __getitem__(self, idx):
        if self.mode == 'train':
            HR_image, LR_image = self.get_hr_lr_pair(idx)
            H, W, C = LR_image.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            # rnd_h = random.randint(0, max(0, H - self.LR_size))
            # rnd_w = random.randint(0, max(0, W - self.LR_size))
            
            mode = random.randint(0, 7)

            # 还是要做数据增强
            LR_image, HR_image = augment_img(LR_image, mode=mode), augment_img(HR_image, mode=mode)
            # DEBUG 这个要改回去
            rnd_h = random.choice(np.arange(0, LR_image.shape[0] - self.LR_size + 1, self.LR_size))
            rnd_w = random.choice(np.arange(0, LR_image.shape[1] - self.LR_size + 1, self.LR_size))            
            LR_image = LR_image[rnd_h:rnd_h + self.LR_size, rnd_w:rnd_w + self.LR_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.scale), int(rnd_w * self.scale)
            HR_image = HR_image[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            
            # 默认数据已经算好了
            if self.LR_path == 'BIC':
                save_path = self.HR_path + '_yadapt_aug_whole_img'
            else:
                save_path = self.LR_path + '_yadapt_aug_whole_img'

            # i 是 patch 的 index
            try:
                # yadapt_feature_path = os.path.join(save_path, os.path.basename(self.LR_images[idx]).split(".")[0]+'_'+str(i)+'_'+str(mode)+'_yadapt.npy')
                yadapt_feature_path = os.path.join(save_path, os.path.basename(self.LR_images[idx]).split(".")[0]+'_'+str(mode)+'_yadapt.npy')
                yadapt_features = np.load(yadapt_feature_path)

                if rnd_w//self.LR_size+self.LR_size // 16 > yadapt_features.shape[2]:
                    print(rnd_w//self.LR_size)
                    print(rnd_w//self.LR_size+self.LR_size // 16)

                # yadapt_features = torch.from_numpy(yadapt_features).float()
                yadapt_features = torch.from_numpy(yadapt_features[:,rnd_h//self.LR_size:rnd_h//self.LR_size+self.LR_size // 16,\
                                                                   rnd_w//self.LR_size:rnd_w//self.LR_size+self.LR_size // 16]).float()

            except:
                # DEBUG 这个要改回去
                print('Error in loading yadapt features')
                yadapt_features = torch.randn(1, 1, 1, 1)

            # DEBUG 看看 yadapt_features 的数据分布
            if DEBUG:
                mean = yadapt_features.mean()
                std = yadapt_features.std()
                avg = yadapt_features.mean(dim=0)
                print('mean: {}, std: {}, avg: {}'.format(mean, std, avg))


            # To numpy
            HR_image = np.array(HR_image) / 255.0
            LR_image = np.array(LR_image) / 255.0
    
            # To tensor
            HR_image = torch.from_numpy(HR_image).permute(2, 0, 1).float()
            LR_image = torch.from_numpy(LR_image).permute(2, 0, 1).float()
    
            return LR_image, HR_image, yadapt_features

        if self.mode == "test":
            HR_image, LR_image = self.get_hr_lr_pair(idx)
            # 需要检查一下一开始能不能对上 HR 和 LR
            try:
                assert HR_image.shape[0] == LR_image.shape[0] * self.scale and HR_image.shape[1] == LR_image.shape[1] * self.scale, "HR and LR should have the same size after modcrop"
            except:
                LR_image = self.force_crop(LR_image, (HR_image.shape[0] // self.scale, HR_image.shape[1] // self.scale))
                print('HR and LR should have the same size after modcrop')
                assert HR_image.shape[0] == LR_image.shape[0] * self.scale and HR_image.shape[1] == LR_image.shape[1] * self.scale, "HR and LR should have the same size after modcrop"


            # assert HR_image.shape[0] == LR_image.shape[0] * self.scale and HR_image.shape[1] == LR_image.shape[1] * self.scale, "HR and LR should have the same size after modcrop"
            HR_image = HR_image[self.overlap:HR_image.shape[0]-self.overlap, self.overlap:HR_image.shape[1] - self.overlap]

            patches, (padded_height, padded_width), (img_height, img_width) = extract_patches(LR_image, self.pretrained_sam_img_size, self.overlap)
            
            batch_LR_image = np.zeros((len(patches), 3, self.pretrained_sam_img_size, self.pretrained_sam_img_size), dtype=np.float32)

            for i, (patch, _, _) in enumerate(patches):
                batch_LR_image[i] = patch.transpose(2, 0, 1)

            if self.LR_path == 'BIC':
                save_path = self.HR_path + '_yadapt_aug_whole_img'
            else:
                save_path = self.LR_path + '_yadapt_aug_whole_img'

            yadapt_feature_path = os.path.join(save_path, os.path.basename(self.LR_images[idx]).split(".")[0]+'_yadapt.npy')
            yadapt_features = np.load(yadapt_feature_path) # [C, h // 16, w // 16]

            batch_yadapt_features = np.zeros(((padded_height*padded_width) // (48*48), yadapt_features.shape[0], 3,3))

            # batch_lr = []
            for i in range(0, padded_height//48, 3):
                for j in range(0, padded_width//48, 3):
                    batch_yadapt_features[i+j] = yadapt_features[:, i:i+3, j:j+3]
                    # batch_lr.append(LR_image[i:i+self.pretrained_sam_img_size, j:j+self.pretrained_sam_img_size])

            # batch_lr = np.array(batch_lr).transpose(0,3,1,2) / 255.0
            # batch_lr = torch.from_numpy(batch_lr).float()
            batch_yadapt_features = torch.from_numpy(batch_yadapt_features).float()
            try:
                assert batch_yadapt_features.shape[0] == batch_LR_image.shape[0], "batch_yadapt_features and batch_LR_image should have the same batch_size"
            except:
                print('batch_yadapt_features and batch_LR_image should have the same batch_size')

            batch_LR_image = batch_LR_image / 255.0
            batch_LR_image = torch.from_numpy(batch_LR_image).float()
            
            return batch_LR_image, HR_image, batch_yadapt_features, patches, (img_height, img_width), (padded_height, padded_width)
            # return batch_lr, batch_yadapt_features, patches, (img_height, img_width), (padded_height, padded_width)

        if self.mode == "pred":
            _, LR_image = self.get_hr_lr_pair(idx)
            img_name = os.path.basename(self.LR_images[idx])
            patches, (padded_height, padded_width), (img_height, img_width) = extract_patches(LR_image, self.pretrained_sam_img_size, self.overlap)
            
            batch_LR_image = np.zeros((len(patches), 3, self.pretrained_sam_img_size, self.pretrained_sam_img_size), dtype=np.float32)
            for i, (patch, _, _) in enumerate(patches):
                batch_LR_image[i] = patch.transpose(2, 0, 1)

            if self.LR_path == 'BIC':
                save_path = self.HR_path + '_yadapt_aug'
            else:
                save_path = self.LR_path + '_yadapt_aug'

            yadapt_feature_path = os.path.join(save_path, os.path.basename(self.LR_images[idx]).split(".")[0]+'_yadapt.npy')
            yadapt_features = np.load(yadapt_feature_path)
            
            batch_yadapt_features = torch.from_numpy(yadapt_features).float()
            assert batch_yadapt_features.shape[0] == batch_LR_image.shape[0], "batch_yadapt_features and batch_LR_image should have the same batch_size"

            batch_LR_image = batch_LR_image / 255.0
            batch_LR_image = torch.from_numpy(batch_LR_image).float()
            return batch_LR_image, batch_yadapt_features, patches, (img_height, img_width), (padded_height, padded_width), img_name


def load_dataset(config_path):
    import sys
    import yaml
    import os
    sys.path.append('/home/mayanze/PycharmProjects/SwinTF')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    test_set = SuperResolutionYadaptDataset(config=config['train'])
    return test_set

def precompute_dataset(config_path):
    test_set = load_dataset(config_path)
    del test_set
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/blur_iso.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()

    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/blur_aniso.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()

    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/jpeg.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()

    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()

    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/degrade.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()


    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/test_config/manga109test.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()

    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()
    
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/test_config/set5.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()

    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/test_config/Set14test.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()

    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/test_config/BSDS100.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()

    # DIV2K = SuperResolutionDataset(config=config['train'])
    # LR_image, HR_image = DIV2K.__getitem__(0)
    # print(LR_image.shape, HR_image.shape)
    from tqdm import tqdm
    config_path = '/home/mayanze/PycharmProjects/SwinTF/config/Set5_ddp.yaml'
    test_set = load_dataset(config_path)
    # iterate all the train set
    for i in tqdm(range(len(test_set))):
        lr, hr, y = test_set.__getitem__(436)
        if y.shape != (3840, 3, 3):
            print(test_set.HR_images[436])
            print(y.shape)
    # print(LR_image.shape, HR_image.shape)

    # DIV2K = SuperResolutionYadaptDataset(config=config['train'])
    # LR_image, HR_image, yadapt = DIV2K.__getitem__(0)
    # print(LR_image.shape, HR_image.shape, yadapt.shape)


    # test_set = SuperResolutionPrecomputeYadaptDataset(config=config['test'])
    # LR_image, HR_image, yadapt, (x,y)= test_set.__getitem__(0)
    # print(LR_image.shape, HR_image.shape, yadapt.shape)

    # test_set = SuperResolutionYadaptDataset(config=config['test'])
    # LR_image, HR_image, yadapt, (x,y)= test_set.__getitem__(0)
    # print(LR_image.shape, HR_image.shape, yadapt.shape)
    # print(test_set.precompute) 
    # precompute(test_set, config)


    # from tqdm import tqdm

    # Set CUDA_VISIBLE_DEVICES environment variable
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # train_set = SuperResolutionYadaptDataset(config=config['train'])
    # # train_set.precompute_train()
    # _, _ ,_ = train_set.__getitem__(0)

    # total_train_set = len(train_set)

    # y1_max = []
    # y2_max = []
    # y3_max = []

    # y1_min = []
    # y2_min = []
    # y3_min = []
    # for i in tqdm(range(total_train_set)):
    #     LR_image, HR_image, yadapt = train_set.__getitem__(i)
    #     # Equally Split the yadapt into three part by 0 dimension
    #     yadapt = yadapt.numpy()
    #     yadapt = np.split(yadapt, 3, axis=0)
    #     y1_max.append(np.max(yadapt[0]))
    #     y2_max.append(np.max(yadapt[1]))
    #     y3_max.append(np.max(yadapt[2]))
    #     y1_min.append(np.min(yadapt[0]))
    #     y2_min.append(np.min(yadapt[1]))
    #     y3_min.append(np.min(yadapt[2]))

    # # Create a plot to show three histograms
    # # Create a histogram
    # import matplotlib.pyplot as plt
    # # A 3x2 plot and Larger figure size
    # fig, axs = plt.subplots(3, 2, figsize=(15, 17))
    # axs[0, 0].hist(y1_max, bins=100)
    # axs[0, 0].set_title('y1_max')
    # axs[0, 1].hist(y1_min, bins=100)
    # axs[0, 1].set_title('y1_min')
    # axs[1, 0].hist(y2_max, bins=100)
    # axs[1, 0].set_title('y2_max')
    # axs[1, 1].hist(y2_min, bins=100)
    # axs[1, 1].set_title('y2_min')
    # axs[2, 0].hist(y3_max, bins=100)
    # axs[2, 0].set_title('y3_max')
    # axs[2, 1].hist(y3_min, bins=100)
    # axs[2, 1].set_title('y3_min')

    # # Save the histogram
    # plt.savefig('yadapt_histogram.png')

    # train_set = SuperResolutionYadaptDataset(config=config['train'])
    # r0_LR_image, r0_HR_image, r0_yadapt = train_set.__getitem__(0)
    # train_set2 = SuperResolutionYadaptDataset(config=config['train'])
    # r1_LR_image, r1_HR_image, r1_yadapt = train_set2.__getitem__(0)

    # # # 检查是否完全一样
    # print(np.allclose(r0_LR_image, r1_LR_image)) # 由于有随机，所以不同是正常的
    # print(np.allclose(r0_HR_image, r1_HR_image)) # 由于有随机，所以不同是正常的
    # print(np.allclose(r0_yadapt, r1_yadapt))

    #=========================================================================
    ### 检查出来确实是不一样，yadapt_features 为什么不一样，继续检查
    #=========================================================================


