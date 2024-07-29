import numpy as np
import cv2



def extract_patches(image, patch_size, overlap):
    stride = patch_size - overlap
    
    # 这里选择padding的方式是constant，所以padding的部分是黑色的
    img_height, img_width = image.shape[:2]
    pad_height = (stride - (img_height - patch_size) % stride) % stride
    pad_width = (stride - (img_width - patch_size) % stride) % stride
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=255)

    # 不能裁剪！！！裁剪了之后图片缩小太多了
    # padded_image = image[:image.shape[0] - (image.shape[0] - patch_size) % stride, :image.shape[1] - (image.shape[1] - patch_size) % stride]

    patches = []
    padded_height, padded_width = padded_image.shape[:2]
    for y in range(0, padded_height - patch_size + 1, stride):
        for x in range(0, padded_width - patch_size + 1, stride):
            patch = padded_image[y:y + patch_size, x:x + patch_size]
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





hr_image = cv2.imread('/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set5/GTmod12/baby.png')
hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)
img = cv2.imread('/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set5/LRbicx2/baby.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img

patch_size = 48
overlap = 6
stride = patch_size - overlap
patches, (padded_height, padded_width), (img_height, img_width) = extract_patches(img, patch_size, overlap)
super_res_patches = super_resolve_patches(patches, scale_factor=2)
super_res_image = overlapping_image(super_res_patches, padded_height, padded_width, patch_size, overlap, stride, scale_factor=2)

# FIXME 这里的计算是否准确实在心里面拿不准
super_res_image = super_res_image[:img_height*2-overlap*2, :img_width*2 - overlap*2]
cut_hr_image = hr_image[overlap:hr_image.shape[0]-overlap, overlap:hr_image.shape[1] - overlap]
# Save the super-resolved image
cv2.imwrite('super_res_image.png', super_res_image)
cv2.imwrite('original_image.png', img[:padded_height, :padded_width])

# Print 损失的图片大小
h_loss = img.shape[0] - padded_height
w_loss = img.shape[1] - padded_width
print(h_loss, w_loss)


# 鉴于现在已经是9个patch，并且有overlap，下面的问题就是思考overlap怎么切的问题
# 粗浅的来看有三种情况
# 1. 在角落的patch，只有两个overlap
# 2. 在边缘的patch，有三个overlap
# 3. 在中间的patch，有四个overlap


