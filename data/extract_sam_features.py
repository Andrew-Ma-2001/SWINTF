# This file is intended to extract sam features from the dataset.
# First load the model, and the pretrained path
# Then load the dataset and go through each image
# Save the features as .npy file

import sys
sys.path.append("/home/mayanze/PycharmProjects/SwinTF/")

from nets.build_sam import ImageEncoderViT_FeatureExtract
from data.data_utils import get_all_images

from functools import partial
import yaml
import torch
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm

def extract_sam_model(model_path = '/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth', image_size = 48):
# Using ViT-h
    encoder_embed_dim=1280
    encoder_depth=32
    encoder_num_heads=16
    encoder_global_attn_indexes=[7, 15, 23, 31]
    # checkpoint=checkpoint
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    image_encoder = ImageEncoderViT_FeatureExtract(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    model_path = model_path
    # image_encoder.load_state_dict(torch.load(model_path), strict=False)

    # Assume `model` is your model and `partial_state_dict` is the state dict you want to load
    model = image_encoder
    # Get the state of the model before loading the new state dict
    before_state = {name: param.clone() for name, param in model.named_parameters()}

    # Load the state dict
    model.load_state_dict(torch.load(model_path), strict=True)

    # Get the state of the model after loading the new state dict
    after_state = {name: param.clone() for name, param in model.named_parameters()}

    # Compare the parameters before and after loading the state dict
    for name, param_before in before_state.items():
        param_after = after_state[name]
        if not torch.all(param_before.eq(param_after)):
            print(f"Parameter {name} was updated.")

    return model


if __name__ == "__main__":
    # config_path = '/home/mayanze/PycharmProjects/SwinTF/config/example copy.yaml'  

    # # Load in the yaml file
    # with open(config_path, 'r') as file:
    #     config = yaml.safe_load(file)

    # # Load in the model
    # model = extract_sam_model()
    # # Move to GPU
    # model = model.cuda()


    # train_config = config['train']
    # test_config = config['test']

    # # Get the train and test images
    # train_images = get_all_images(train_config['train_LR'])
    # test_images = get_all_images(test_config['test_LR'])

    # # Get the train and test features
    # train_features = []
    # test_features = []

    # # Go through each image
    # for i in tqdm(range(len(train_images))):
    #     # Load in the image
    #     image = np.array(Image.open(train_images[i]))
    #     # Reszie the image to 1024x1024 using cv2 BICUBIC
    #     # XXX Using 这里假如用了 Resize 原则上在数据读入的时候就需要？Dataloader？
    #     image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    #     image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    #     # Move to GPU
    #     image = image.cuda()

    #     with torch.no_grad():
    #         # Get the feature
    #         _, x10, x20, x30 = model(image)
    #         feature = [x10.squeeze(0).cpu().numpy(), x20.squeeze(0).cpu().numpy(), x30.squeeze(0).cpu().numpy()]

    #     # Make feature a numpy array
    #     feature = np.array(feature)
    #     # Print the shape
    #     print(feature.shape)
    #     # Save the feature
    #     train_features.append(feature)

    # # Save the train features
    # np.save('trained_sam_features.npy', train_features)


    # # Go through each image
    # for i in tqdm(range(len(test_images))):
    #     # Load in the image
    #     image = np.array(Image.open(test_images[i]))
    #     # Reszie the image to 1024x1024 using cv2 BICUBIC
    #     # XXX Using 这里假如用了 Resize 原则上在数据读入的时候就需要？Dataloader？
    #     image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    #     image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    #     # Move to GPU
    #     image = image.cuda()

    #     with torch.no_grad():
    #         # Get the feature
    #         _, x10, x20, x30 = model(image)
    #         feature = [x10.squeeze(0).cpu().numpy(), x20.squeeze(0).cpu().numpy(), x30.squeeze(0).cpu().numpy()]

    #     # Make feature a numpy array
    #     feature = np.array(feature)
    #     # Print the shape
    #     print(feature.shape)
    #     # Save the feature
    #     test_features.append(feature)

    # # Save the train features
    # np.save('tested_sam_features.npy', test_features)

    model = extract_sam_model(image_size=1024)

