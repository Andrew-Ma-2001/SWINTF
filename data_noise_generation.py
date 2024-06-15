import os
import torch
import numpy as np
from noise import *
from utils.utils_image import imresize_np, modcrop, augment_img
from data.dataloader import load_dataset
# Define 两个 function imread 和 imsave
from PIL import Image

def imread(path):
    # Use PIL to read an image from the specified path and return it as a numpy array
    return np.array(Image.open(path))

def imsave(path, img):
    # Convert numpy array to PIL Image and save it to the specified path
    img = np.clip(img, 0, 1)
    img = img*255.0
    img = img.astype(np.uint8)
    img_pil = Image.fromarray(img)
    img_pil.save(path)


####

# image_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109'
# bic_save_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/lr'
# scale_factor = 2

# if not os.path.exists(bic_save_dir):
#     os.makedirs(bic_save_dir)

# # 先批量把 bic 的图像生成到文件夹中
# for file in os.listdir(image_dir):
#     img_path = os.path.join(image_dir, file)
#     if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#         img = imread(img_path)
#         img = modcrop(img, scale_factor)
#         bic_img = imresize_np(img/255.0, 1/scale_factor, True)
#         imsave(os.path.join(bic_save_dir, file), bic_img)
#     else:
#         print(f"Skipping non-image file: {file}")

# image_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109'
# bic_save_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/hr'
# scale_factor = 2

# if not os.path.exists(bic_save_dir):
#     os.makedirs(bic_save_dir)

# for file in os.listdir(image_dir):
#     img_path = os.path.join(image_dir, file)
#     if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#         img = imread(img_path)
#         img = modcrop(img, scale_factor)
#         # bic_img = imresize_np(img/255.0, 1/scale_factor, True)
#         imsave(os.path.join(bic_save_dir, file), img /255.0)
#     else:
#         print(f"Skipping non-image file: {file}")

# 然后针对几个 noise 类型生成对应的图像
# input_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/lr'
# output_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/blur_iso'
# output_folder_aniso = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/blur_aniso'
# output_folder_jpeg = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/jpeg'
# output_folder_noise = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/noise'
# output_folder_degrade = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/degrade'

# scale_factor = 2  # or 4

# process_images(input_folder, output_folder, apply_isotropic_gaussian_blur, scale_factor)
# process_images(input_folder, output_folder_aniso, apply_anisotropic_gaussian_blur, scale_factor)
# process_images(input_folder, output_folder_jpeg, add_jpeg_noise, scale_factor)
# process_images(input_folder, output_folder_noise, add_gaussian_noise, scale_factor)
# process_images(input_folder, output_folder_degrade, degrade_image, scale_factor)


config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/blur_iso.yaml'
test_set = load_dataset(config_path)
test_set = None
torch.cuda.empty_cache()
config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/blur_aniso.yaml'
test_set = load_dataset(config_path)
test_set = None
torch.cuda.empty_cache()
config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/jpeg.yaml'
test_set = load_dataset(config_path)
test_set = None
torch.cuda.empty_cache()
config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise.yaml'
test_set = load_dataset(config_path)
test_set = None
torch.cuda.empty_cache()
config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/degrade.yaml'
test_set = load_dataset(config_path)
test_set = None