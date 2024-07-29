import sys

sys.path.append("..")
from nets.build_sam import sam_model_registry
from nets.predictor import SamPredictor
from nets.automatic_mask_generator import SamAutomaticMaskGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image


def preprocess_image(image: np.ndarray, device) -> torch.Tensor:
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    final_shape = 1024
    # Deep copy the image
    img = np.copy(image)
    w, h = image.shape[1], image.shape[0]
    scale = final_shape / max(w, h)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    img = np.array(resize(to_pil_image(img), (new_h, new_w)))
    img = torch.tensor(img, device=device).permute(2, 0, 1).contiguous()[None, :, :, :]
    img = (img - pixel_mean) / pixel_std
    pad_w = final_shape - new_w
    pad_h = final_shape - new_h
    img = F.pad(img, (0, pad_w, 0, pad_h))
    return img


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

# predictor = SamPredictor(sam)

image = cv2.imread('/Users/mayanze/Desktop/SWINTF-master-20240726/nets/0001x2.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = preprocess_image(image, 'cpu')

# Save the img as a npy file
np.save('img.npy', img.numpy())
masks = mask_generator.generate(image)
# predictor.set_image(image)
# masks = mask_generator_2.generate(image)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()

# # Gerate a random image
# import numpy as np
#
#
# # # Genreate a 0-255 image
# # image = np.random.randint(0, 255, (500, 300, 3)).astype(np.uint8)
#
#
# from PIL import Image
# image = Image.open('0001x2.png')
# image = np.array(image)[:48,:48,:]
# large_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
# large_img[:48,:48,:] = image
# plt.imshow(large_img)
# plt.show()


# predictor.set_image(image)
# masks, scores, logits = predictor.predict(
#     # point_coords=input_point,
#     # point_labels=input_label,
#     multimask_output=True,
# )
#
# a = 1