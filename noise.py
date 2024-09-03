import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import rotate
from PIL import Image, ImageFilter
import random
from main_test_swinir import test_main, define_model
from tqdm import tqdm
import yaml


def apply_isotropic_gaussian_blur(image, scale_factor):
    kernel_size = random.choice(range(7, 22, 2))
    if scale_factor == 2:
        sigma = np.random.uniform(0.1, 2.4)
    elif scale_factor == 4:
        sigma = np.random.uniform(0.1, 2.8)
    else:
        raise ValueError("Unsupported scale_factor. Use 2 or 4.")
    
    truncate = kernel_size / (2 * sigma)
    
    # Apply Gaussian blur to each channel separately
    blurred_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        blurred_image[:, :, i] = gaussian_filter(image[:, :, i], sigma=sigma, mode='reflect', truncate=truncate)
    
    return blurred_image


def apply_anisotropic_gaussian_blur(image, scale_factor):
    kernel_size = random.choice(range(7, 22, 2))
    theta = np.random.uniform(0, np.pi)
    if scale_factor == 2:
        sigma_x = np.random.uniform(0.5, 6)
        sigma_y = np.random.uniform(0.5, 6)
    elif scale_factor == 4:
        sigma_x = np.random.uniform(0.5, 8)
        sigma_y = np.random.uniform(0.5, 8)
    else:
        raise ValueError("Unsupported scale_factor. Use 2 or 4.")
    
    truncate_x = kernel_size / (2 * sigma_x)
    truncate_y = kernel_size / (2 * sigma_y)
    
    # Apply anisotropic Gaussian blur to each channel separately
    blurred_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        blurred_image_x = gaussian_filter(image[:, :, i], sigma=sigma_x, mode='reflect', truncate=truncate_x)
        blurred_image_rot = rotate(blurred_image_x, theta, reshape=False, mode='reflect')
        blurred_image_y = gaussian_filter(blurred_image_rot, sigma=sigma_y, mode='reflect', truncate=truncate_y)
        blurred_image[:, :, i] = rotate(blurred_image_y, -theta, reshape=False, mode='reflect')
    
    return blurred_image


def add_gaussian_noise(image, scale_factor):
    sigma = random.uniform(1/255, 25/255)
    noise_type = random.choices(['general', 'channel', 'gray'], [0.2, 0.4, 0.4])[0]
    
    if noise_type == 'general':
        cov_matrix = np.random.randn(3, 3)
        cov_matrix = cov_matrix @ cov_matrix.T  # Make it symmetric positive definite
        noise = np.random.multivariate_normal(np.zeros(3), cov_matrix, image.shape[:2])
    elif noise_type == 'channel':
        noise = np.random.normal(0, sigma, image.shape)
    elif noise_type == 'gray':
        noise = np.random.normal(0, sigma, (image.shape[0], image.shape[1]))
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def add_gaussian_noise_with_params(image, scale_factor, sigma, noise_type, cov_matrix=None):
    """
    Usage Example:
    sigma = 0.1
    noise_type = 'channel'
    noisy_image = add_gaussian_noise_with_params(image, scale_factor, sigma, noise_type)
    """
    if noise_type == 'general':
        if cov_matrix is None:
            cov_matrix = np.random.randn(3, 3)
            cov_matrix = cov_matrix @ cov_matrix.T  # Make it symmetric positive definite
        noise = np.random.multivariate_normal(np.zeros(3), cov_matrix, image.shape[:2])
        noise = noise*sigma
    elif noise_type == 'channel':
        noise = np.random.normal(0, sigma, image.shape)
    elif noise_type == 'gray':
        noise = np.random.normal(0, sigma, (image.shape[0], image.shape[1]))
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    else:
        raise ValueError("Unsupported noise_type. Use 'general', 'channel', or 'gray'.")
    
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def generate_gaussian_noise_dir_with_params(input_folder, output_folder, sigma, noise_type, cov_matrix=None, scale_factor=2):
    """
    Generate Gaussian noise for each image in the input folder and save it to the output folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate Gaussian noise for each image in the input folder and save it to the output folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            image = np.asarray(Image.open(input_path).convert('RGB')) / 255.0
            noisy_image = add_gaussian_noise_with_params(image, scale_factor, sigma, noise_type, cov_matrix)
            Image.fromarray((noisy_image * 255).astype(np.uint8)).save(output_path)
            print(f"Processed and saved: {output_path}")


def apply_jpeg_compression(image, quality):
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    image_pil = image_pil.filter(ImageFilter.GaussianBlur(0))  # To ensure PIL don't optimize away the compression
    image_pil.save('temp.jpg', 'JPEG', quality=quality)
    compressed_image = Image.open('temp.jpg')
    return np.asarray(compressed_image) / 255.0

def add_jpeg_noise(image, scale_factor):
    quality = random.randint(30, 95)
    image = apply_jpeg_compression(image, quality)
    return image    


def degrade_image(image, scale_factor):
    # Apply isotropic Gaussian blur
    image = apply_isotropic_gaussian_blur(image, scale_factor)

    # Apply anisotropic Gaussian blur
    image = apply_anisotropic_gaussian_blur(image, scale_factor)

    # Add Gaussian noise
    image = add_gaussian_noise(image, scale_factor)

    # Apply JPEG compression twice
    if random.random() < 0.75:
        quality = random.randint(30, 95)
        image = apply_jpeg_compression(image, quality)
    
    quality = random.randint(30, 95)
    image = apply_jpeg_compression(image, quality)
    
    return image


def process_images(input_folder, output_folder, process_function, scale_factor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):  #DEBUG only process 10 images
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Load image
            image = np.asarray(Image.open(input_path).convert('RGB')) / 255.0
            
            # Process image
            processed_image = process_function(image, scale_factor)
            
            # Save processed image
            Image.fromarray((processed_image * 255).astype(np.uint8)).save(output_path)
            print(f"Processed and saved: {output_path}")

###### 2024-08-12 ######
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def uint2single(img):
    return np.float32(img/255.)

def add_JPEG_noise(img, quality_factor):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cal_sigma(sig_x, sig_y, radians):
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)

    D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
    U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                   torch.cat([radians.sin(), radians.cos()], 2)], 1)
    sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))

    return sigma


def anisotropic_gaussian_kernel(batch, kernel_size, covar):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    xy = torch.stack([xx, yy], -1).view(batch, -1, 2)

    inverse_sigma = torch.inverse(covar)
    kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, kernel_size, kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)


def isotropic_gaussian_kernel(batch, kernel_size, sigma):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size//2
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

    return kernel / kernel.sum([1,2], keepdim=True)


def random_anisotropic_gaussian_kernel(batch=1, kernel_size=21, lambda_min=0.2, lambda_max=4.0):
    theta = torch.rand(batch).cuda() * math.pi
    lambda_1 = torch.rand(batch).cuda() * (lambda_max - lambda_min) + lambda_min
    lambda_2 = torch.rand(batch).cuda() * (lambda_max - lambda_min) + lambda_min

    covar = cal_sigma(lambda_1, lambda_2, theta)
    kernel = anisotropic_gaussian_kernel(batch, kernel_size, covar)
    return kernel


def stable_anisotropic_gaussian_kernel(kernel_size=21, theta=0, lambda_1=0.2, lambda_2=4.0):
    theta = torch.ones(1).cuda() * theta / 180 * math.pi
    lambda_1 = torch.ones(1).cuda() * lambda_1
    lambda_2 = torch.ones(1).cuda() * lambda_2

    covar = cal_sigma(lambda_1, lambda_2, theta)
    kernel = anisotropic_gaussian_kernel(1, kernel_size, covar)
    return kernel


def random_isotropic_gaussian_kernel(batch=1, kernel_size=21, sig_min=0.2, sig_max=4.0):
    x = torch.rand(batch).cuda() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(batch, kernel_size, x)
    return k


def stable_isotropic_gaussian_kernel(kernel_size=21, sig=4.0):
    x = torch.ones(1).cuda() * sig
    k = isotropic_gaussian_kernel(1, kernel_size, x)
    return k


def random_gaussian_kernel(batch, kernel_size=21, blur_type='iso_gaussian', sig_min=0.2, sig_max=4.0, lambda_min=0.2, lambda_max=4.0):
    if blur_type == 'iso_gaussian':
        return random_isotropic_gaussian_kernel(batch=batch, kernel_size=kernel_size, sig_min=sig_min, sig_max=sig_max)
    elif blur_type == 'aniso_gaussian':
        return random_anisotropic_gaussian_kernel(batch=batch, kernel_size=kernel_size, lambda_min=lambda_min, lambda_max=lambda_max)


def stable_gaussian_kernel(kernel_size=21, blur_type='iso_gaussian', sig=2.6, lambda_1=0.2, lambda_2=4.0, theta=0):
    if blur_type == 'iso_gaussian':
        return stable_isotropic_gaussian_kernel(kernel_size=kernel_size, sig=sig)
    elif blur_type == 'aniso_gaussian':
        return stable_anisotropic_gaussian_kernel(kernel_size=kernel_size, lambda_1=lambda_1, lambda_2=lambda_2, theta=theta)

# implementation of matlab bicubic interpolation in pytorch
class bicubic(nn.Module):
    def __init__(self):
        super(bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32).cuda()
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32).cuda()

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0), torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1), torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1/4):
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0]
        weight1 = weight1[0]

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out


class Gaussin_Kernel(object):
    def __init__(self, kernel_size=21, blur_type='iso_gaussian',
                 sig=2.6, sig_min=0.2, sig_max=4.0,
                 lambda_1=0.2, lambda_2=4.0, theta=0, lambda_min=0.2, lambda_max=4.0):
        self.kernel_size = kernel_size
        self.blur_type = blur_type

        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.theta = theta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def __call__(self, batch, random):
        # random kernel
        if random == True:
            return random_gaussian_kernel(batch, kernel_size=self.kernel_size, blur_type=self.blur_type,
                                          sig_min=self.sig_min, sig_max=self.sig_max,
                                          lambda_min=self.lambda_min, lambda_max=self.lambda_max)

        # stable kernel
        else:
            return stable_gaussian_kernel(kernel_size=self.kernel_size, blur_type=self.blur_type,
                                          sig=self.sig,
                                          lambda_1=self.lambda_1, lambda_2=self.lambda_2, theta=self.theta)

class BatchBlur(nn.Module):
    def __init__(self, kernel_size=21):
        super(BatchBlur, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        else:
            self.pad = nn.ReflectionPad2d((kernel_size//2, kernel_size//2-1, kernel_size//2, kernel_size//2-1))

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        input = input.float()
        input_pad = self.pad(input)
        H_p, W_p = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
            kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
        else:
            input_CBHW = input_pad.view((1, C * B, H_p, W_p))
            kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
            kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, groups=B*C).view((B, C, H, W))

class SRMDPreprocessing(object):
    def __init__(self,
                 scale,
                 mode='bicubic',
                 kernel_size=21,
                 blur_type='iso_gaussian',  # 一般各向同性
                 sig=2.6,  # 参数一
                 sig_min=0.2,
                 sig_max=4.0,
                 lambda_1=0.2,
                 lambda_2=4.0,
                 theta=0,
                 lambda_min=0.2,
                 lambda_max=4.0,
                 noise=0.0 # 高斯强度 参数二
                 ):
        '''
        # sig, sig_min and sig_max are used for isotropic Gaussian blurs
        During training phase (random=True):
            the width of the blur kernel is randomly selected from [sig_min, sig_max]
        During test phase (random=False):
            the width of the blur kernel is set to sig

        # lambda_1, lambda_2, theta, lambda_min and lambda_max are used for anisotropic Gaussian blurs
        During training phase (random=True):
            the eigenvalues of the covariance is randomly selected from [lambda_min, lambda_max]
            the angle value is randomly selected from [0, pi]
        During test phase (random=False):
            the eigenvalues of the covariance are set to lambda_1 and lambda_2
            the angle value is set to theta
        '''
        self.kernel_size = kernel_size
        self.scale = scale
        self.mode = mode
        self.noise = noise

        self.gen_kernel = Gaussin_Kernel(
            kernel_size=kernel_size, blur_type=blur_type,
            sig=sig, sig_min=sig_min, sig_max=sig_max,
            lambda_1=lambda_1, lambda_2=lambda_2, theta=theta, lambda_min=lambda_min, lambda_max=lambda_max
        )
        self.blur = BatchBlur(kernel_size=kernel_size)
        self.bicubic = bicubic()

    def __call__(self, hr_tensor, random=True):
        with torch.no_grad():
            # only downsampling
            if self.gen_kernel.blur_type == 'iso_gaussian' and self.gen_kernel.sig == 0:
                B, C, H, W = hr_tensor.size()
                # hr_blured = hr_tensor.view(B, C, H, W)
                hr_blured = hr_tensor
                b_kernels = None

            # gaussian blur + downsampling
            else:
                B, C, H, W = hr_tensor.size()
                b_kernels = self.gen_kernel(B, random)  # B degradations

                # blur
                hr_blured = self.blur(hr_tensor, b_kernels)
                # hr_blured = hr_blured.view(-1, C, H, W)  # BN, C, H, W

            # downsampling
            if self.mode == 'bicubic':
                lr_blured = self.bicubic(hr_blured, scale=1/self.scale)
            elif self.mode == 's-fold':
                lr_blured = hr_blured.view(-1, C, H//self.scale, self.scale, W//self.scale, self.scale)[:, :, :, 0, :, 0]
                # lr_blured = hr_blured
                raise NotImplementedError("s-fold downsampling is not implemented")

            # add noise
            if self.noise > 0:
                _, C, H_lr, W_lr = lr_blured.size()
                noise_level = torch.rand(B, 1, 1, 1, 1).to(lr_blured.device) * self.noise if random else self.noise
                noise = torch.randn_like(lr_blured).view(-1, C, H_lr, W_lr).mul_(noise_level).view(-1, C, H_lr, W_lr)
                lr_blured.add_(noise)

            lr_blured = torch.clamp(lr_blured.round(), 0, 255)


            # return lr_blured.view(B, N, C, H//int(self.scale), W//int(self.scale)), b_kernels
            return lr_blured, b_kernels

def main_noise_processing(config, scale_factor, sig, noise, quality_factor):
    preprocess = SRMDPreprocessing(scale=scale_factor, sig=sig, noise=noise)

    # Do processing in the images in the folder
    input_folder = config['test']['test_HR']
    output_folder = os.path.join(config['test']['test_LR'], f'noise_sig_{sig}_noise_{noise}_quality_{quality_factor}')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            # Read in the image and change to tensor
            image = np.asarray(Image.open(input_path).convert('RGB'))
            image = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
            image = image.cuda()
            # Process the image
            lr_blured, b_kernels = preprocess(image, random=False)

            # Save the processed image
            lr_blured = lr_blured.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Do JPEG compression
            if quality_factor != 0:
                lr_noisy = add_JPEG_noise(lr_blured, quality_factor)
                Image.fromarray((lr_noisy).astype(np.uint8)).save(output_path)
            else:
                Image.fromarray((lr_blured).astype(np.uint8)).save(output_path)

    del preprocess
    
    return input_folder, output_folder

def process_single_image(image, scale_factor, sig, noise, quality_factor):
    preprocess = SRMDPreprocessing(scale=scale_factor, sig=sig, noise=noise)

    # Convert image to tensor
    image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).cuda()

    # Process the image
    lr_blured, b_kernels = preprocess(image_tensor, random=False)

    # Convert processed image back to numpy array
    lr_blured = lr_blured.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Apply JPEG compression if quality_factor is not 0
    if quality_factor != 0:
        lr_noisy = add_JPEG_noise(lr_blured, quality_factor)
        processed_image = lr_noisy
    else:
        processed_image = lr_blured

    return processed_image


def noise_and_test(config_path, model, noise_settings, test_swinir):
    base_config = config_path

    # DEBUG
    # output_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_0.0_quality_50'
    # Load in the config
    with open(base_config, 'r') as f:
        base_config = yaml.safe_load(f)

    input_folder_list = []
    output_folder_list = []
    for noise_set in noise_settings:
        sig, noise, quality_factor = noise_set['sig'], noise_set['noise'], noise_set['quality_factor']
        input_folder, output_folder = main_noise_processing(base_config, base_config['test']['scale'], sig, noise, quality_factor)
        # base_config['test']['test_LR'] = output_folder
        # avg_psnr = test_main(config_path, model, test_swinir)
        # print(f"Average PSNR for noise_set {noise_set}: {avg_psnr}")
        input_folder_list.append(input_folder)
        output_folder_list.append(output_folder)
    
    return input_folder_list, output_folder_list
# Example usage

# Example usage
if __name__ == "__main__":
    # input_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109'
    # output_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/blur_iso'
    # output_folder_aniso = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/blur_aniso'
    # output_folder_jpeg = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/jpeg'
    # output_folder_noise = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/noise'
    # output_folder_degrade = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/degrade'

    scale_factor = 2  # or 4
    
    # # process_images(input_folder, output_folder, apply_isotropic_gaussian_blur, scale_factor)
    # # process_images(input_folder, output_folder_aniso, apply_anisotropic_gaussian_blur, scale_factor)
    # # process_images(input_folder, output_folder_jpeg, add_jpeg_noise, scale_factor)
    # # process_images(input_folder, output_folder_noise, add_gaussian_noise, scale_factor)
    # # process_images(input_folder, output_folder_degrade, degrade_image, scale_factor)
    # # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/test_config/Set14test.yaml'
    config_path = '/home/mayanze/PycharmProjects/SwinTF/config/X2/Set14test.yaml'
    # model_path = 'experiments/SwinIR_20240803080852/500000_model.pth'
    model_path = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'
    test_swinir = True
    noise_settings = [
        # {'sig': 0.0, 'noise': 0.0, 'quality_factor': 30},
        # {'sig': 0.0, 'noise': 0.0, 'quality_factor': 40},
        # {'sig': 0.0, 'noise': 0.0, 'quality_factor': 50},
        # {'sig': 0.0, 'noise': 0.0, 'quality_factor': 60},
        # {'sig': 0.0, 'noise': 0.0, 'quality_factor': 70},
        # {'sig': 0.0, 'noise': 0.0, 'quality_factor': 80},
        # {'sig': 0.0, 'noise': 0.0, 'quality_factor': 90},
        # {'sig': 0.0, 'noise': 0.0, 'quality_factor': 100},
        # {'sig': 0.0, 'noise': 0.0, 'quality_factor': 0},
        # # Testing Sig
        # {'sig': 1.0, 'noise': 0.0, 'quality_factor': 0},
        # {'sig': 2.0, 'noise': 0.0, 'quality_factor': 0},
        # {'sig': 3.0, 'noise': 0.0, 'quality_factor': 0},
        # {'sig': 4.0, 'noise': 0.0, 'quality_factor': 0},
        # {'sig': 5.0, 'noise': 0.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 1.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 2.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 3.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 4.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 5.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 10.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 20.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 30.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 40.0, 'quality_factor': 0},
        # {'sig': 0.0, 'noise': 50.0, 'quality_factor': 0},
        {'sig': 3.0, 'noise': 25.0, 'quality_factor': 50},
    ]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    scale = config['train']['scale']
    training_patch_size = config['train']['patch_size'] // scale
    # tile = training_patch_size
    tile = None
    tile_overlap = 32
    config['test']['tile'] = tile
    config['test']['tile_overlap'] = tile_overlap
    config['network']['swinir_test'] = test_swinir
    model = define_model(scale, training_patch_size, model_path, config)

    if test_swinir:
        model = model.cuda()
    
    config_name_list = []
    _, output_folder_list = noise_and_test(config_path, model, noise_settings, test_swinir)
    for output_folder, noise_set in zip(output_folder_list, noise_settings):
        sig, noise, quality_factor = noise_set['sig'], noise_set['noise'], noise_set['quality_factor']
        config_name = f'{output_folder}_noise_{sig}_noise_{noise}_quality_{quality_factor}.yaml'
        config_name_list.append(config_name)
        config['test']['test_LR'] = output_folder
        with open(config_name, 'w') as f:
            yaml.dump(config, f)
    
    print(config_name_list)
    

    # # Load your image here
    # image = np.asarray(Image.open('/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_test/hr/img_020.png').convert('RGB')) / 255.0
    # scale_factor = 2  # or 4
    # # Test the functionality of each degradation function on a sample image
    # print("Testing isotropic Gaussian blur...")
    # blurred_image_iso = apply_isotropic_gaussian_blur(image, scale_factor)
    # Image.fromarray((blurred_image_iso * 255).astype(np.uint8)).save('output/blurred_image_iso.jpg')

    # print("Testing anisotropic Gaussian blur...")
    # blurred_image_aniso = apply_anisotropic_gaussian_blur(image, scale_factor)
    # Image.fromarray((blurred_image_aniso * 255).astype(np.uint8)).save('output/blurred_image_aniso.jpg')

    # print("Testing addition of Gaussian noise...")
    # noisy_image = add_gaussian_noise(image, scale_factor)
    # Image.fromarray((noisy_image * 255).astype(np.uint8)).save('output/noisy_image.jpg')

    # print("Testing JPEG compression...")
    # compressed_image = apply_jpeg_compression(image, 50)  # Using a fixed quality of 50 for testing
    # Image.fromarray((compressed_image * 255).astype(np.uint8)).save('output/compressed_image.jpg')

    # print("All degradation functions tested successfully.")

    # degraded_image = degrade_image(image, scale_factor)
    # Image.fromarray((degraded_image * 255).astype(np.uint8)).save('output/degraded_image.jpg')

    # Create a list of sigma
    # sigma = np.linspace(0, 15, 10)
    # sigma = sigma / 100
    # for s in sigma:
    #     new_img = add_gaussian_noise_with_params(image, scale_factor, s, 'general')
    #     # Image.fromarray((new_img * 255).astype(np.uint8)).save(f'output/new_img_{s}.jpg')
    #     # Save the original and noisy images to the output directory
    #     output_dir = 'output'
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     # Save the original image
    #     original_image_path = os.path.join(output_dir, 'original_image.jpg')
    #     Image.fromarray((image * 255).astype(np.uint8)).save(original_image_path)

    #     # Save the noisy image
    #     noisy_image_path = os.path.join(output_dir, f'noisy_image_sigma_{s:.6f}.jpg')
    #     Image.fromarray((new_img * 255).astype(np.uint8)).save(noisy_image_path)


    # image = np.asarray(Image.open('/home/mayanze/PycharmProjects/SwinTF/nets/0901x2.png').convert('RGB'))
    # scale_factor = 1
    # sig = 5
    # noise = 10
    # quality_factor = 80
    # processed_image = process_single_image(image, scale_factor, sig, noise, quality_factor)
    # Image.fromarray((processed_image).astype(np.uint8)).save('processed_image.jpg')