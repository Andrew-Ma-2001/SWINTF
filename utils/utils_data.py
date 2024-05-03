import os
import cv2
import torch
import numpy as np


def _get_all_images(path):
    """
    Get all images from a path.
    """
    # Write a warning if the path is not exist
    if not os.path.exists(path):
        raise ValueError('Path [{:s}] not exists.'.format(path))

    images = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                images.append(os.path.join(root, file))

    # Sort the images
    images = sorted(images)
    return images

def get_all_images(path):
    paths = None
    if isinstance(path, str):
        paths = _get_all_images(path)
    elif isinstance(path, list):
        paths = []
        for p in path:
            paths += _get_all_images(p)
    else:
        raise ValueError('Wrong path type: [{:s}].'.format(type(path)))
    return paths

# Function to split a batch into smaller sub-batches
def split_batch(batch, max_sub_batch_size):
    sub_batches = []
    for start in range(0, batch.size(0), max_sub_batch_size):
        end = min(start + max_sub_batch_size, batch.size(0))
        sub_batches.append(batch[start:end])
    return sub_batches

# Process the batch across multiple GPUs with DataParallel
def process_batch(batch, model, max_sub_batch_size):
    # Split the batch into sub-batches that fit into GPU memory
    sub_batches = split_batch(batch, max_sub_batch_size)

    # Process each sub-batch using DataParallel and collect the results
    output_batches = []
    y1_ = []
    y2_ = []
    y3_ = []
    for sub_batch in sub_batches:
        sub_batch = sub_batch.cuda() # Move sub-batch to default GPU device before using DataParallel
        with torch.no_grad():
            _, y1, y2, y3 = model(sub_batch)
        # Move output to CPU to avoid GPU memory accumulation
        y1_.append(y1.cpu())
        y2_.append(y2.cpu())
        y3_.append(y3.cpu())
    
    y1 = torch.cat(y1_, dim=0)
    y2 = torch.cat(y2_, dim=0)
    y3 = torch.cat(y3_, dim=0)

    return y1, y2, y3


def extract_patches(image, patch_size, overlap):
    stride = patch_size - overlap
    
    # 这里选择padding的方式是constant，所以padding的部分是黑色的
    img_height, img_width = image.shape[:2]
    pad_height = (stride - (img_height - patch_size) % stride) % stride
    pad_width = (stride - (img_width - patch_size) % stride) % stride
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0,0)), mode='constant', constant_values=255)

    # 不能裁剪！！！裁剪了之后图片缩小太多了
    # padded_image = image[:image.shape[0] - (image.shape[0] - patch_size) % stride, :image.shape[1] - (image.shape[1] - patch_size) % stride]

    patches = []
    padded_height, padded_width = padded_image.shape[:2]
    for y in range(0, padded_height - patch_size + 1, stride):
        for x in range(0, padded_width - patch_size + 1, stride):
            patch = padded_image[y:y + patch_size, x:x + patch_size,:]
            patches.append((patch, x, y))
    
    return patches, padded_image.shape[:2], image.shape[:2]


def super_resolve_patch(patch, scale_factor=2):
    super_res_patch = cv2.resize(patch, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    return super_res_patch


def super_resolve_patches(patches, scale_factor=2):
    super_res_patches = []
    for (patch, x, y) in patches:
        super_res_patch = super_resolve_patch(patch, scale_factor)
        super_res_patches.append((super_res_patch, x, y))
    return super_res_patches


def overlapping_image(patches, padded_height, padded_width, patch_size, overlap, stride, scale_factor):
    """
    >>> img = cv2.imread('/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/BSDS100/8023.png')
    >>> img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    >>> img = img[:132, :132]
    >>> patch_size = 48
    >>> overlap = 4
    >>> stride = patch_size - overlap
    >>> patches, (padded_height, padded_width) = extract_patches(img, patch_size, overlap)
    >>> super_res_patches = super_resolve_patches(patches, scale_factor=2)
    >>> super_res_image = overlapping_image(super_res_patches, padded_height, padded_width, patch_size, overlap, stride, scale_factor=2)
    """
    max_num_y = padded_height // stride
    max_num_x = padded_width // stride
    
    super_res_image = np.zeros((stride * (padded_height // stride) * scale_factor, stride * (padded_width // stride) * scale_factor), dtype=np.uint8)
    for (patch, x, y) in patches:
        # x_pos = x * scale_factor
        # y_pos = y * scale_factor
        # 这样写的代价是还要裁掉周围一圈的overlap，等于在HR图像上丢失了一圈像素，但是这样我不会写了，所以还是分成三种情况
        num_y = y // stride
        num_x = x // stride

        # TODO 对于patch需要考虑scale_factor的问题

        # # 对于在角落的patch，只有两个overlap
        # if [num_y, num_x] in [[0, 0], [0, max_num_x-1], [max_num_y-1, 0], [max_num_y-1, max_num_x-1]]:
        #     if num_y == 0 and num_x == 0:
        #         patch = patch[:(patch_size-overlap//2)*scale_factor, :(patch_size-overlap//2)*scale_factor]
        #         # super_res_image[:(patch_size-overlap//2)*scale_factor, :(patch_size-overlap//2)*scale_factor] = patch
        #     elif num_y == 0 and num_x == max_num_x-1:
        #         patch = patch[(overlap//2)*scale_factor:, :(patch_size-overlap//2)*scale_factor] # yx
        #         # super_res_image[:(patch_size-overlap//2)*scale_factor, num_x*stride*scale_factor:num_x*stride*scale_factor+stride*scale_factor] = patch
        #     elif num_y == max_num_y-1 and num_x == 0:
        #         patch = patch[:(patch_size-overlap//2)*scale_factor, (overlap//2)*scale_factor:]
        #         # super_res_image[num_y*stride*scale_factor:num_y*stride*scale_factor+stride*scale_factor, :(patch_size-overlap//2)*scale_factor] = patch
        #     elif num_y == max_num_y-1 and num_x == max_num_x-1:
        #         patch = patch[(overlap//2)*scale_factor:, (overlap//2)*scale_factor:]
        #         # super_res_image[num_y*stride*scale_factor:num_y*stride*scale_factor+stride*scale_factor, num_x*stride*scale_factor:num_x*stride*scale_factor+stride*scale_factor] = patch

        # # 对于在边缘的patch，有三个overlap
        # elif num_y == 0 or num_x == 0 or num_y == max_num_y-1 or num_x == max_num_x-1:
        #     if num_y == 0:
        #         patch = patch[:(patch_size-overlap//2)*scale_factor, (overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor]
        #     elif num_x == 0:
        #         patch = patch[(overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor, :(patch_size-overlap//2)*scale_factor]
        #     elif num_y == max_num_y-1:
        #         patch = patch[(overlap//2)*scale_factor:, (overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor]
        #     elif num_x == max_num_x-1:
        #         patch = patch[(overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor, (overlap//2)*scale_factor:]
        
        # # 对于在中间的patch，有四个overlap
        # else:
        #     patch = patch[(overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor, (overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor]
        patch = patch[(overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor, (overlap//2)*scale_factor:(patch_size-overlap//2)*scale_factor]
        super_res_image[num_y*stride*scale_factor:(num_y+1)*stride*scale_factor, num_x*stride*scale_factor:(num_x+1)*stride*scale_factor] = patch
    return super_res_image

