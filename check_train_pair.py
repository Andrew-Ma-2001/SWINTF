from data.dataloader import SuperResolutionYadaptDataset
import yaml
import os
import cv2

config = '/home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml'
with open(config, 'r') as f:
    cfg = yaml.safe_load(f)

train_set = SuperResolutionYadaptDataset(config=cfg['train'])

LR_image, HR_image, _ = train_set.__getitem__(0)
LR_image = LR_image.permute(1, 2, 0).numpy() * 255
HR_image = HR_image.permute(1, 2, 0).numpy() * 255

cv2.imwrite('LR_image.png', LR_image)
cv2.imwrite('HR_image.png', HR_image)
