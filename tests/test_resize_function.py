import os
import sys
from PIL import Image
import numpy as np

sys.path.append("/home/mayanze/PycharmProjects/SwinTF/")
from utils.utils_image import *

gt_img = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set14/LRbicx2/baboon.png'
resize_img = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set14/GTmod12/baboon.png'
mt_img = '/home/mayanze/PycharmProjects/SwinTF/tests/baboon_resized.png'
scale = 2

def load_img(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    return img

gt = load_img(gt_img)
resize = load_img(resize_img)
mt = load_img(mt_img)

print(gt.shape)
print(resize.shape)

resize_img = imresize_np(resize/255.0, 1/scale)
print(resize_img.shape)

# see if resize img are the same to gt
print(np.all((resize_img*255).astype(np.float32) == gt))

print(np.all(mt == gt))
# print out the difference and show it in a figure
diff = np.abs((resize_img*255).astype(int) - gt.astype(int))
print(diff.shape)

import matplotlib.pyplot as plt
plt.imshow(diff.astype(np.uint8), cmap='gray')
plt.savefig('diff.png')