import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import rotate
from PIL import Image, ImageFilter
import random


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

# Example usage
if __name__ == "__main__":
    input_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109'
    output_folder = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/blur_iso'
    output_folder_aniso = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/blur_aniso'
    output_folder_jpeg = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/jpeg'
    output_folder_noise = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/noise'
    output_folder_degrade = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109_test/degrade'

    scale_factor = 2  # or 4
    
    process_images(input_folder, output_folder, apply_isotropic_gaussian_blur, scale_factor)
    process_images(input_folder, output_folder_aniso, apply_anisotropic_gaussian_blur, scale_factor)
    process_images(input_folder, output_folder_jpeg, add_jpeg_noise, scale_factor)
    process_images(input_folder, output_folder_noise, add_gaussian_noise, scale_factor)
    process_images(input_folder, output_folder_degrade, degrade_image, scale_factor)



    # Load your image here
    image = np.asarray(Image.open('/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set5/LRbicx2/bird.png').convert('RGB')) / 255.0
    scale_factor = 2  # or 4
    # Test the functionality of each degradation function on a sample image
    print("Testing isotropic Gaussian blur...")
    blurred_image_iso = apply_isotropic_gaussian_blur(image, scale_factor)
    Image.fromarray((blurred_image_iso * 255).astype(np.uint8)).save('output/blurred_image_iso.jpg')

    print("Testing anisotropic Gaussian blur...")
    blurred_image_aniso = apply_anisotropic_gaussian_blur(image, scale_factor)
    Image.fromarray((blurred_image_aniso * 255).astype(np.uint8)).save('output/blurred_image_aniso.jpg')

    print("Testing addition of Gaussian noise...")
    noisy_image = add_gaussian_noise(image, scale_factor)
    Image.fromarray((noisy_image * 255).astype(np.uint8)).save('output/noisy_image.jpg')

    print("Testing JPEG compression...")
    compressed_image = apply_jpeg_compression(image, 50)  # Using a fixed quality of 50 for testing
    Image.fromarray((compressed_image * 255).astype(np.uint8)).save('output/compressed_image.jpg')

    print("All degradation functions tested successfully.")

    degraded_image = degrade_image(image, scale_factor)
    Image.fromarray((degraded_image * 255).astype(np.uint8)).save('output/degraded_image.jpg')