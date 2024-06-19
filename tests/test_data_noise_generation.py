import unittest
import os
import sys
sys.path.append('/home/mayanze/PycharmProjects/SwinTF')
import shutil
import numpy as np
from PIL import Image
from data_noise_generation import process_images_hr_lr, process_noise

class TestProcessImagesHrLr(unittest.TestCase):
    def setUp(self):
        self.image_dir = "/tmp/test_images"
        self.lr_save_dir = "/tmp/test_images_lr"
        self.hr_save_dir = "/tmp/test_images_hr"
        self.scale_factor = 2
        os.makedirs(self.image_dir, exist_ok=True)
        # Create a sample image
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img.save(os.path.join(self.image_dir, "test.jpg"))

    def tearDown(self):
        shutil.rmtree(self.image_dir)
        shutil.rmtree(self.lr_save_dir, ignore_errors=True)
        shutil.rmtree(self.hr_save_dir, ignore_errors=True)

    def test_process_images_scale_factor(self):
        process_images_hr_lr(self.image_dir, self.lr_save_dir, self.hr_save_dir, self.scale_factor)
        hr_img_path = os.path.join(self.hr_save_dir, "test.jpg")
        lr_img_path = os.path.join(self.lr_save_dir, "test.jpg")
        with Image.open(hr_img_path) as hr_img, Image.open(lr_img_path) as lr_img:
            hr_width, hr_height = hr_img.size
            lr_width, lr_height = lr_img.size
            self.assertEqual(hr_width, lr_width * self.scale_factor)
            self.assertEqual(hr_height, lr_height * self.scale_factor)

    def test_process_images(self):
        result = process_images_hr_lr(self.image_dir, self.lr_save_dir, self.hr_save_dir, self.scale_factor)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.lr_save_dir))
        self.assertTrue(os.path.exists(self.hr_save_dir))
        self.assertTrue(os.path.exists(os.path.join(self.lr_save_dir, "test.jpg")))
        self.assertTrue(os.path.exists(os.path.join(self.hr_save_dir, "test.jpg")))

    def test_process_images_exception(self):
        # Test with a non-existent image directory
        result = process_images_hr_lr("/non/existent/dir", self.lr_save_dir, self.hr_save_dir, self.scale_factor)
        self.assertFalse(result)

# TODO 这个测试用例有问题，需要修改
class TestProcessNoise(unittest.TestCase):
    def setUp(self):
        self.input_folder = "/tmp/test_images"
        self.hr_folder = "/tmp/test_images_hr"
        self.config_save_path = "/tmp/test_configs"
        self.sigmas = [0.1, 0.2]
        self.noise_types = ["general"]
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.hr_folder, exist_ok=True)
        os.makedirs(self.config_save_path, exist_ok=True)
        # Create a sample image
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img.save(os.path.join(self.input_folder, "test.jpg"))
    
    def tearDown(self):
        shutil.rmtree(self.input_folder)
        shutil.rmtree(self.hr_folder)
        shutil.rmtree(self.config_save_path)

    def test_process_noise(self):
        result = process_noise(self.input_folder, self.hr_folder, self.config_save_path, self.sigmas, self.noise_types)
        self.assertTrue(result)
        for sigma in self.sigmas:
            for noise_type in self.noise_types:
                sigma_str = str(int(sigma*255)).zfill(2)
                output_folder = os.path.join(self.input_folder, f"noise_{sigma_str}_{noise_type}")
                config_name = f"noise_{sigma_str}_{noise_type}.yaml"
                self.assertTrue(os.path.exists(output_folder))
                self.assertTrue(os.path.exists(os.path.join(self.config_save_path, config_name)))

    def test_process_noise_exception(self):
        # Test with a non-existent input directory
        result = process_noise("/non/existent/dir", self.hr_folder, self.config_save_path, self.sigmas, self.noise_types)
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()