import math
import torch
import numpy as np
import os
import sys
sys.path.append('/home/mayanze/PycharmProjects/SwinTF/')
from PIL import Image
from utils.utils_data import get_all_images
from data.extract_sam_features import extract_sam_model
from utils.utils_image import augment_img, imresize_np

class ImagePreprocessor:
    def __init__(self, sam_img_size=1024):
        self.sam_img_size = sam_img_size
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).cuda()
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).cuda()

    def augment_img(self, image, mode):
        return augment_img(image, mode)
    
    def set_augment(self, augment):
        self.augment = augment

    def set_mode(self, mode):
        self.mode = mode

    def set_image(self, image):
        self.image = image
        self.w, self.h = image.shape[1], image.shape[0]
        self.target_size = self.get_img_target_size(image)
        self.pad_img = self.mirror_padding(image, self.target_size)

    def mirror_padding(self, image, target_size):
        # Pad the image to the target size
        pad_w = target_size[0] - image.shape[1]
        pad_h = target_size[1] - image.shape[0]
        return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')



    def get_img_target_size(self, image):
        w, h = image.shape[1], image.shape[0]
        scale = self.sam_img_size / max(w, h)

        if scale < 1:
            upper_bound = math.ceil(max(w, h) / self.sam_img_size)
            target_size = (self.sam_img_size * upper_bound, self.sam_img_size * upper_bound)
        else:
            target_size = (self.sam_img_size, self.sam_img_size)

        return target_size

    def create_patches(self, image, target_size):
        patches = []
        for i in range(0, target_size[0], 1024):
            for j in range(0, target_size[1], 1024):
                patches.append(image[j:j + 1024, i:i + 1024])
        return patches

    def reshape_patches(self, patches):
        img = np.zeros((self.target_size[1], self.target_size[0], 3))
        for i, patch in enumerate(patches):
            j = i // (self.target_size[0] // 1024)
            k = i % (self.target_size[0] // 1024)
            img[j*1024:j*1024+1024, k*1024:k*1024+1024, :] = patch
        return img

    def reshape_yadapt_feature(self, total_yadapt_feature):
        # Now i have a [x, 3840, 64, 64] change it to [3840, 64*x, 64*x]
        x = total_yadapt_feature.shape[0]
        large_img = np.zeros((total_yadapt_feature.shape[1], 64*x, 64*x))
        for i in range(x):
            large_img[:, i*64:i*64+64, i*64:i*64+64] = total_yadapt_feature[i]
        return large_img

    def preprocess_image_v2(self, device) -> torch.Tensor:
        if self.target_size[0] > 1024:
            # Slice the image into 1024x1024 patches
            patches = self.create_patches(self.pad_img, self.target_size)
            tensor_img = torch.tensor(np.array(patches), device=device).permute(0, 3, 1, 2).contiguous()
            # Check if image range is 0-1 or 0-255
            if tensor_img.max() <= 1.0:
                tensor_img = tensor_img * 255.0
            tensor_img = (tensor_img - self.pixel_mean) / self.pixel_std
            return tensor_img
        else:
            img = torch.tensor(self.pad_img, device=device).permute(2, 0, 1).contiguous()[None, :, :, :]
            # Check if image range is 0-1 or 0-255 
            if img.max() <= 1.0:
                img = img * 255.0
            img = (img - self.pixel_mean) / self.pixel_std
            return img


    def slice_yadapt_features(self, total_yadapt_feature):
        large_yadapt_feature = self.reshape_yadapt_feature(total_yadapt_feature)
        cut_yadapt_feature = large_yadapt_feature[:, 0:math.ceil(self.h/16)+1,0:math.ceil(self.w/16)+1]
        return cut_yadapt_feature

    def clear_image(self):
        self.image = None
        self.w = None
        self.h = None
        self.target_size = None
        self.pad_img = None

def train_precompute(use_vit=False, scale=2):
    os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
    LR_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic/X'+str(scale)
    LR_size = 48
    pretrained_sam_img_size = 48
    use_cuda = True
    save_path = LR_path + '_yadapt_aug_whole_img_vit' if use_vit else LR_path + '_yadapt_aug_whole_img'  #XXX 改回来
    model = extract_sam_model(model_path='/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth', image_size = 1024)
    model = model.cuda()
    model.image_encoder = torch.nn.DataParallel(model.image_encoder)
    preprocessor = ImagePreprocessor()
    LR_images = get_all_images(LR_path)
    # LR_images = ['dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic/X2/0437x2.png']

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert len(os.listdir(save_path)) == 0, "The save_path should be empty"

    modes = [0,1,2,3,4,5,6,7] # 8 modes

    with torch.no_grad():
        for mode in modes:
            for idx in range(len(LR_images)):
                LR_image = np.array(Image.open(LR_images[idx]))
                LR_image = augment_img(LR_image, mode)
                preprocessor.set_image(LR_image)
                torch_img = preprocessor.preprocess_image_v2(device='cuda')

                if use_vit:
                    x, _, _, _ = model.image_encoder(torch_img)
                    y = x
                else:
                    _, y1, y2, y3 = model.image_encoder(torch_img)
                    # Concate y1 y2 y3 by torch
                    y = torch.cat([y1, y2, y3], dim=1)
                y = y.cpu().numpy()
                # breakpoint()
                y = preprocessor.slice_yadapt_features(y)
                # Save y to the save_path
                save_name = os.path.join(save_path, os.path.basename(LR_images[idx]).split(".")[0]+'_'+str(mode)+'_yadapt.npy')
                np.save(save_name, y)
                # breakpoint()
                print('Save {}'.format(save_name))
                preprocessor.clear_image()

            



def test_precompute():
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    LR_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109'
    BIC = True
    LR_size = 48
    pretrained_sam_img_size = 48
    use_cuda = True
    save_path = LR_path + '_yadapt_aug_whole_img' 
    model = extract_sam_model(model_path='/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth', image_size = 1024)
    model = model.cuda()
    model.image_encoder = torch.nn.DataParallel(model.image_encoder)
    preprocessor = ImagePreprocessor()
    LR_images = get_all_images(LR_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert len(os.listdir(save_path)) == 0, "The save_path should be empty"


    with torch.no_grad():
        for idx in range(len(LR_images)):
            LR_image = np.array(Image.open(LR_images[idx]))


            if BIC:
                LR_image = imresize_np(LR_image/255.0, 1/2)
                LR_image = LR_image * 255.0


            preprocessor.set_image(LR_image)
            torch_img = preprocessor.preprocess_image_v2(device='cuda')
            _, y1, y2, y3 = model.image_encoder(torch_img)
            # Concate y1 y2 y3 by torch
            y = torch.cat([y1, y2, y3], dim=1)
            y = y.cpu().numpy()
            y = preprocessor.slice_yadapt_features(y)
            # Save y to the save_path
            save_name = os.path.join(save_path, os.path.basename(LR_images[idx]).split(".")[0]+'_yadapt.npy')
            np.save(save_name, y)
            print('Save {}'.format(save_name))
            preprocessor.clear_image()



# 把预处理函数写在外面，不能写在里面了
def check_train_precompute():
    import numpy as np
    import torch
    import sys
    import os
    sys.path.append('/home/mayanze/PycharmProjects/SwinTF/')
    from PIL import Image
    from utils.utils_data import get_all_images, process_batch
    from data.extract_sam_features import extract_sam_model
    from utils.utils_image import augment_img

    LR_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic/X2'
    LR_size = 48
    pretrained_sam_img_size = 48
    use_cuda = True
    save_path = LR_path + '_yadapt_aug' 
    model = extract_sam_model(model_path='/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth', image_size = 1024)
    # only use 0,1 gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]

    pixel_mean = np.array(pixel_mean)
    pixel_std = np.array(pixel_std)

    if use_cuda:
        model = model.cuda()
        model.image_encoder = torch.nn.DataParallel(model.image_encoder)

    LR_images = get_all_images(LR_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #assert len(os.listdir(save_path)) == 0, "The save_path should be empty"

    modes = [0,1,2,3,4,5,6,7] # 8 modes

    for mode in modes:
        for idx in range(len(LR_images)):
            LR_image = Image.open(LR_images[idx])
            LR_image = np.array(LR_image)
            patches = []

            LR_image = (LR_image - pixel_mean) / pixel_std
            # Cut the LR_image into patches
            # for y in range(0, LR_image.shape[0] - LR_size + 1, LR_size):
            #     for x in range(0, LR_image.shape[1] - LR_size + 1, LR_size):
            #         patch = LR_image[y:y + LR_size, x:x + LR_size,:]
            #         patch = augment_img(patch, mode)
            #         patches.append((patch, x, y))

            # batch_LR_image = np.zeros((len(patches), 3, pretrained_sam_img_size, pretrained_sam_img_size))
            # for i, (patch, _, _) in enumerate(patches):
                # batch_LR_image[i] = patch.transpose(2, 0, 1)

            # 这里要把 48x48 变成 1024x1024 建一个更大的矩阵
            LR_image = LR_image.transpose(2, 0, 1)
            large_img = np.zeros([1, 3, 1024, 1024])
            large_img[0, :, :LR_image.shape[1], :LR_image.shape[2]] = LR_image
            # 然后将 batch_LR_image 转换成 tensor
            batch_LR_image_sam = torch.from_numpy(large_img).float()
            # 然后将 batch_LR_image 输入到模型中
            inferece_batch_size = 1

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
            # for i in range(yadapt_features.shape[0]):
            #     save_name = os.path.join(save_path, os.path.basename(LR_images[idx]).split(".")[0]+'_'+str(i)+'_'+str(mode)+'_yadapt.npy')
            #     np.save(save_name, yadapt_features[i])
            #     print('Save {}'.format(save_name))

            # DEBUG
            # 对于 yadapt_features 直接保存 
            save_name = os.path.join(save_path, os.path.basename(LR_images[idx]).split(".")[0]+'_yadapt.npy')
            np.save(save_name, yadapt_features)
            print('Save {}'.format(save_name))

# check_precompute()

# 这里 test 是要计算 overlap 的

def check_test_precompute():
    # 这里有两个注意的点，一个是有的要BIC，有的不用BIC，我的提议是两个分开写
    # 那这里就是写不用BIC的
    import numpy as np
    import torch
    import sys
    import os
    sys.path.append('/home/mayanze/PycharmProjects/SwinTF/')
    from PIL import Image
    from utils.utils_data import get_all_images, process_batch
    from data.extract_sam_features import extract_sam_model
    from utils.utils_image import imresize_np, modcrop
    from utils.utils_data import extract_patches

    LR_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/BSDS100'
    LR_size = 48
    pretrained_sam_img_size = 48
    use_cuda = True
    save_path = LR_path + '_yadapt_aug'
    model = extract_sam_model(model_path='/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth', image_size = 1024)
    # only use 0,1 gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,4,5'

    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]

    pixel_mean = np.array(pixel_mean)
    pixel_std = np.array(pixel_std)

    if use_cuda:
        model = model.cuda()
        model.image_encoder = torch.nn.DataParallel(model.image_encoder)

    LR_images = get_all_images(LR_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert len(os.listdir(save_path)) == 0, "The save_path should be empty"

    for idx in range(len(LR_images)):
        LR_image = Image.open(LR_images[idx])
        LR_image = modcrop(LR_image, 2)
        LR_image = np.array(LR_image)
        patches = []
        LR_image = (LR_image - pixel_mean) / pixel_std
        overlap = 8

        patches, _, _ = extract_patches(LR_image, LR_size, overlap) 

        batch_LR_image = np.zeros((len(patches), 3, pretrained_sam_img_size, pretrained_sam_img_size))
        for i, (patch, _, _) in enumerate(patches):
            batch_LR_image[i] = patch.transpose(2, 0, 1)
        # 这里要把 48x48 变成 1024x1024 建一个更大的矩阵
        large_img = np.zeros((batch_LR_image.shape[0], 3, 1024, 1024))
        large_img[:, :, :48, :48] = batch_LR_image
        # 然后将 batch_LR_image 转换成 tensor
        batch_LR_image_sam = torch.from_numpy(large_img).float()
        # 然后将 batch_LR_image 输入到模型中
        inferece_batch_size = 15
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
        # for i in range(yadapt_features.shape[0]):
        #     save_name = os.path.join(save_path, os.path.basename(LR_images[idx]).split(".")[0]+'_'+str(i)+'_yadapt.npy')
        #     np.save(save_name, yadapt_features[i])
        #     print('Save {}'.format(save_name))

        # 对于 yadapt_features 直接保存
        save_name = os.path.join(save_path, os.path.basename(LR_images[idx]).split(".")[0]+'_yadapt.npy')
        np.save(save_name, yadapt_features)
        print('Save {}'.format(save_name))

# check_test_precompute()



if __name__ == '__main__':
    # test_precompute()
    train_precompute(use_vit=True)
#     from PIL import Image
#     # check_train_precompute()
#     preprocessor = ImagePreprocessor()
#     # test create pathces and reshape patches
#     img = Image.open('/home/mayanze/PycharmProjects/SwinTF/nets/0001x2.png')
#     img = np.array(img)
#     preprocessor.set_image(img)
#     image = preprocessor.mirror_padding(img, preprocessor.target_size)
#     patches = preprocessor.create_patches(image, preprocessor.target_size)
#     print(patches[0].shape)
#     reshaped_patches = preprocessor.reshape_patches(patches)
#     print(reshaped_patches.shape)
#     Image.fromarray(reshaped_patches.astype(np.uint8)).save('test.png')