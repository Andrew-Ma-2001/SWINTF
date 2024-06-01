import os
from PIL import Image
import torch
import numpy as np
from nets.swinir import SwinIRAdapter
from utils.utils_data import get_all_images, extract_patches, process_batch
from tqdm import tqdm
from data.extract_sam_features import extract_sam_model
import yaml
import matplotlib.pyplot as plt


#DEBUG
config_path = '/home/mayanze/PycharmProjects/SwinTF/config/test_config/aim2019overlapzero.yaml'
print('Config path: {}'.format(config_path))
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


model_path= '/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240503_113731/230000_model.pth'
gpu_ids='0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
# save_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019_lr/'  # 必须有 / 结尾

scale = config['test']['scale']

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

checkpoint = torch.load(model_path)

model.eval()
model.cuda()
model = torch.nn.DataParallel(model) 
    
    
model.load_state_dict(checkpoint,strict=True)
print('Resume from checkpoint from {}'.format(model_path))

# Calculate the yadapt for the image while cutting into patches
sam_model = extract_sam_model(model_path=config['test']['pretrained_sam'], image_size = 1024)
sam_model.eval()
sam_model.cuda()
sam_model.image_encoder = torch.nn.DataParallel(sam_model.image_encoder)


overlap = 0
patch_size = config['test']['pretrained_sam_img_size']
stride = patch_size - overlap
scale_factor = 2
LR_image_paths = get_all_images(config['test']['test_LR'])



save_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/aim2019/aim2019_overlap0_x4/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def predict_one_img(LR_image_path, img_name, model, sam_model, overlap, patch_size, stride, scale_factor, save_path):
    LR_image = Image.open(LR_image_path).convert('RGB')
    LR_image = np.array(LR_image)


    pixel_mean = np.array([123.675, 116.28, 103.53])
    pixel_std = np.array([58.395, 57.12, 57.375])

    sam_image = LR_image
    sam_image = np.array(sam_image)
    sam_image = (sam_image - pixel_mean) / pixel_std

    lr_patches, _, _ = extract_patches(LR_image/255.0, patch_size, overlap, constant_values=0.5)
    lr_batch_LR_image = np.zeros((len(lr_patches), 3, patch_size, patch_size), dtype=np.float32)
    for i, (patch, _, _) in enumerate(lr_patches):
        lr_batch_LR_image[i] = patch.transpose(2, 0, 1)
    lr_batch_LR_image = torch.from_numpy(lr_batch_LR_image).float()

    sam_patches, (padded_height, padded_width), (img_height, img_width) = extract_patches(sam_image, patch_size, overlap, constant_values=0.5)
    sam_batch_LR_image = np.zeros((len(sam_patches), 3, patch_size, patch_size), dtype=np.float32)
    for i, (patch, _, _) in enumerate(sam_patches):
        sam_batch_LR_image[i] = patch.transpose(2, 0, 1)

    # 这里要把 48x48 变成 1024x1024 建一个更大的矩阵
    large_img = np.zeros((sam_batch_LR_image.shape[0], 3, 1024, 1024))
    large_img[:, :, :48, :48] = sam_batch_LR_image
    # 然后将 batch_LR_image 转换成 tensor
    sam_batch_LR_image_sam = torch.from_numpy(large_img).float()
    # 然后将 batch_LR_image 输入到模型中

    with torch.no_grad():
        if sam_batch_LR_image_sam.shape[0] <= 5:
            sam_batch_LR_image_sam = sam_batch_LR_image_sam.cuda()
            _, y1, y2, y3 = sam_model.image_encoder(sam_batch_LR_image_sam)
            y1, y2, y3 = y1.cpu().numpy(), y2.cpu().numpy(), y3.cpu().numpy()
        else:
            y1, y2, y3 = process_batch(sam_batch_LR_image_sam, sam_model.image_encoder, 5)
            # import matplotlib.pyplot as plt
            # plt.imshow(batch_LR_image[0,0,:,:])
            # plt.savefig('test.png')
            # Concatenate the features
        sam_batch_LR_image_sam = sam_batch_LR_image_sam.cpu()
        y1, y2, y3 = y1[:, :, :3, :3], y2[:, :, :3, :3], y3[:, :, :3, :3]
        yadapt_features = np.concatenate((y1, y2, y3), axis=1)
        batch_yadapt_features = torch.from_numpy(yadapt_features).float()
        assert batch_yadapt_features.shape[0] == lr_batch_LR_image.shape[0], "batch_yadapt_features and batch_LR_image should have the same batch_size"

        batch_size = lr_batch_LR_image.size()[0]
        split_size = 100
        batch_Pre_image_parts = []

        with torch.no_grad():
            for i in range(0, batch_size, split_size):
                batch_LR_image_part = lr_batch_LR_image[i:i + split_size].cuda()
                batch_yadapt_features_part = batch_yadapt_features[i:i + split_size].cuda()
                batch_Pre_image_part = model(batch_LR_image_part, batch_yadapt_features_part)
                batch_Pre_image_parts.append(batch_Pre_image_part)
                torch.cuda.empty_cache()

        batch_Pre_image = torch.cat(batch_Pre_image_parts, dim=0)

        batch_Pre_image = batch_Pre_image.clamp(0, 1).cpu().detach().permute(0, 2, 3, 1).numpy()
        batch_Pre_image = batch_Pre_image * 255
        batch_Pre_image = batch_Pre_image.astype(np.uint8)
        super_res_image = np.zeros((stride * (padded_height // stride) * scale_factor, stride * (padded_width // stride) * scale_factor, 3), dtype=np.uint8)
        for (patch, x, y), pre_image in zip(lr_patches, batch_Pre_image):
            num_y = y // stride
            num_x = x // stride
            patch = pre_image[(overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor, (overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor, :]
            super_res_image[num_y*stride*scale_factor:(num_y+1)*stride*scale_factor, num_x*stride*scale_factor:(num_x+1)*stride*scale_factor, :] = patch
        super_res_image = super_res_image[:img_height*scale_factor-overlap*scale_factor, :img_width*scale_factor - overlap*scale_factor]
        save_path = os.path.join(save_path, '{}'.format(img_name))
        plt.imsave(save_path, super_res_image.astype(np.uint8))  #不能这么写，保存图像名字应该一致



for LR_image_path in tqdm(LR_image_paths):
    img_name = os.path.basename(LR_image_path)
    predict_one_img(LR_image_path, img_name, model, sam_model, overlap, patch_size, stride, scale_factor, save_path)
