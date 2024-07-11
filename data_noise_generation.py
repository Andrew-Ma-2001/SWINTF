import os
import torch
import yaml
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


def return_base_config():
    base_config_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/scale2_base.yaml'  #TODO 换成一个别的文件夹下面
    # Read the config from the base config file
    with open(base_config_path, 'r') as file:
        base_config = yaml.safe_load(file)
    return base_config


def generate_config_and_save(LR_dir, HR_dir, save_path, config_name):
    """
    Generate a config file for the test set and save it to the specified path.
    """
    base_config = return_base_config()
    base_config['test']['test_LR'] = LR_dir
    base_config['test']['test_HR'] = HR_dir
    with open(os.path.join(save_path, config_name), 'w') as file:
        yaml.dump(base_config, file)
    print(f"Config saved to: {os.path.join(save_path, config_name)}")
    return base_config

def process_images_hr_lr(image_dir, lr_save_dir, hr_save_dir, scale_factor):
    try:
        if not os.path.exists(lr_save_dir):
            os.makedirs(lr_save_dir)

        # 先批量把 bic 的图像生成到文件夹中
        for file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, file)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img = imread(img_path)
                img = modcrop(img, scale_factor)
                bic_img = imresize_np(img/255.0, 1/scale_factor, True)
                imsave(os.path.join(lr_save_dir, file), bic_img)
            else:
                print(f"Skipping non-image file: {file}")

        if not os.path.exists(hr_save_dir):
            os.makedirs(hr_save_dir)

        for file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, file)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img = imread(img_path)
                img = modcrop(img, scale_factor)
                imsave(os.path.join(hr_save_dir, file), img /255.0)
            else:
                print(f"Skipping non-image file: {file}")

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def process_noise(input_folder, output_folder, hr_folder, config_save_path, sigmas, noise_types):
    try:
        for sigma in sigmas:
            for noise_type in noise_types:
                sigma_str = f'sigma_{sigma:.3f}'.replace('.','_')
                output_folder = os.path.append(output_folder, f'noise_{sigma_str}_{noise_type}')
                config_name = f'noise_{sigma_str}_{noise_type}.yaml'
                generate_gaussian_noise_dir_with_params(input_folder, output_folder, sigma, noise_type, None)
                generate_config_and_save(LR_dir=output_folder, HR_dir=hr_folder, save_path=config_save_path, config_name=config_name)

        configs = [os.path.join(config_save_path, file) for file in os.listdir(config_save_path) if file.endswith('.yaml')]

        for config in configs:
            print(config)
            test_set = load_dataset(config)
            test_set = None
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def generate_previous_noise_dataset_v1():
    # ================================ Manga 109 生成噪声图像 ================================
    # 先批量把 bic 的图像生成到文件夹中

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


    # =============================== Urban100 生成噪声图像 ===============================
    # 先批量把 bic 的图像生成到文件夹中

    # image_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100'
    # bic_save_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_test/lr'
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

    # image_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100'
    # bic_save_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_test/hr'
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

    # # 然后针对几个 noise 类型生成对应的图像
    # input_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_test/lr'
    # output_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_test/blur_iso'
    # output_folder_aniso = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_test/blur_aniso'
    # output_folder_jpeg = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_test/jpeg'
    # output_folder_noise = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_test/noise'
    # output_folder_degrade = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_test/degrade'

    # scale_factor = 2  # or 4

    # process_images(input_folder, output_folder, apply_isotropic_gaussian_blur, scale_factor)
    # process_images(input_folder, output_folder_aniso, apply_anisotropic_gaussian_blur, scale_factor)
    # process_images(input_folder, output_folder_jpeg, add_jpeg_noise, scale_factor)
    # process_images(input_folder, output_folder_noise, add_gaussian_noise, scale_factor)
    # process_images(input_folder, output_folder_degrade, degrade_image, scale_factor)


    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/blur_iso.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/blur_aniso.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/jpeg.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/noise.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    # torch.cuda.empty_cache()
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/degrade.yaml'
    # test_set = load_dataset(config_path)
    # test_set = None
    pass

def generate_previous_noise_dataset_v2():
    #============ Manga109 生成噪声图像 ============
    lr_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_noise/lr'
    hr_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_noise/hr'

    input_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_noise/lr'
    hr_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_noise/hr'
    config_save_path = '/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise'

    # 针对 general noise 重新生成，这次生成的图片会把老的覆盖掉
    # 但是因为老的强度太高了，所以还行，但是要把 general noise 的yadapt重新算一下
    # length = 5
    # # sigmas = np.linspace(1, 15, length) / 255.0
    # sigmas = np.array([1,5,10,15]) / 100.0
    # noise_types = ['general']
    # configs = []

    # for sigma in sigmas:
    #     for noise_type in noise_types:
    #         sigma_str = f'sigma_{sigma:.3f}'.replace('.','_')
    #         output_folder = f'/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_noise/noise_{sigma_str}_{noise_type}'
    #         config_name = f'noise_{sigma_str}_{noise_type}.yaml'
    #         generate_gaussian_noise_dir_with_params(input_folder, output_folder, sigma, noise_type, None)
    #         generate_config_and_save(LR_dir=output_folder, HR_dir=hr_folder, save_path=config_save_path, config_name=config_name)
    #         configs.append(os.path.join(config_save_path, config_name))

    # # configs = [os.path.join(config_save_path, file) for file in os.listdir(config_save_path) if file.endswith('.yaml')]
    # for config in configs:
    #     print(config)
    #     test_set = load_dataset(config)
    #     test_set = None
    #     torch.cuda.empty_cache()
    pass

#====== Urban100 生成噪声图像 ======
image_dir = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100'
lr_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_noise/lr'
hr_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_noise/hr'

process_images_hr_lr(image_dir=image_dir, 
                     lr_save_dir=lr_folder, 
                     hr_save_dir=hr_folder, 
                     scale_factor=2)

input_folder = lr_folder
hr_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_noise/hr'
config_save_path = '/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/noise'
# 针对 general noise 重新生成，这次生成的图片会把老的覆盖掉
length = 5
# sigmas = np.linspace(1, 15, length) / 255.0
sigmas = np.array([1,5,10,15]) / 100.0
noise_types = ['general']
configs = []
for sigma in sigmas:
    for noise_type in noise_types:
        sigma_str = f'sigma_{sigma:.3f}'.replace('.','_')
        output_folder = f'/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_noise/noise_{sigma_str}_{noise_type}'
        config_name = f'noise_{sigma_str}_{noise_type}.yaml'
        generate_gaussian_noise_dir_with_params(input_folder, output_folder, sigma, noise_type, None)
        generate_config_and_save(LR_dir=output_folder, HR_dir=hr_folder, save_path=config_save_path, config_name=config_name)
        configs.append(os.path.join(config_save_path, config_name))
# configs = [os.path.join(config_save_path, file) for file in os.listdir(config_save_path) if file.endswith('.yaml')]
for config in configs:
    print(config)
    test_set = load_dataset(config)
    test_set = None
    torch.cuda.empty_cache()


length = 5
# sigmas = np.linspace(1, 15, length) / 255.0
sigmas = np.array([1,5,10,15]) / 255.0
noise_types = ['gray', 'channel']
configs = []
for sigma in sigmas:
    for noise_type in noise_types:
        sigma_str = str(int(sigma*255)).zfill(2)
        output_folder = f'/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_noise/noise_{sigma_str}_{noise_type}'
        config_name = f'noise_{sigma_str}_{noise_type}.yaml'
        generate_gaussian_noise_dir_with_params(input_folder, output_folder, sigma, noise_type, None)
        generate_config_and_save(LR_dir=output_folder, HR_dir=hr_folder, save_path=config_save_path, config_name=config_name)
        configs.append(os.path.join(config_save_path, config_name))
# configs = [os.path.join(config_save_path, file) for file in os.listdir(config_save_path) if file.endswith('.yaml')]
for config in configs:
    print(config)
    test_set = load_dataset(config)
    test_set = None
    torch.cuda.empty_cache()