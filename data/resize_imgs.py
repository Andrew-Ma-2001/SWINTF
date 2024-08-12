import sys
sys.path.append('/home/mayanze/PycharmProjects/SwinTF/')
import os
import numpy as np
import cv2
from PIL import Image
from utils.utils_image import imresize_np




def generate_lr_imgs(dir, scale):
    lr_dir = f'/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/{os.path.basename(dir)}_lrx{scale}'
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
    
    imgs = os.listdir(dir)
    imgs = [img for img in imgs if img.endswith('.png')]

    for img in imgs:
        img_path = os.path.join(dir, img)
        img_lr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        img_lr = imresize_np(img_lr, 1/scale)

        if img_lr.max() > 1:
            # Clamp the values to [0, 1]
            img_lr = np.clip(img_lr, 0, 1)

        img_lr = (img_lr * 255).round().clip(0, 255).astype(np.uint8)
        img_lr = Image.fromarray(img_lr)
        img_lr.save(os.path.join(lr_dir, img), format='PNG')    
    print(f'Generated {len(imgs)} LR images in {lr_dir}')


dirs = [
    '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100',
    '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/BSDS100',
    '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/manga109'
]
scale = 4


for dir in dirs:
    generate_lr_imgs(dir, scale)