# This file is intended to extract sam features from the dataset.
# First load the model, and the pretrained path
# Then load the dataset and go through each image
# Save the features as .npy file

import sys
sys.path.append("/home/mayanze/PycharmProjects/SwinTF/")

from nets.build_sam import ImageEncoderViT_FeatureExtract, ImageEncoderViT


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

    model = torch.nn.Module()
    model.image_encoder = ImageEncoderViT_FeatureExtract(
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


    # Get the state of the model before loading the new state dict
    # before_state = {name: param.clone() for name, param in model.named_parameters()}

    if model_path:
        # Load the state dict
        model.load_state_dict(torch.load(model_path), strict=False)

    # # Get the state of the model after loading the new state dict
    # after_state = {name: param.clone() for name, param in model.named_parameters()}

    # # Compare the parameters before and after loading the state dict
    # for name, param_before in before_state.items():
    #     param_after = after_state[name]
    #     if not torch.all(param_before.eq(param_after)):
    #         print(f"Parameter {name} was updated.")

    return model

def extract_sam_model_vit(model_path = '/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth', image_size = 48):
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

    model = torch.nn.Module()
    model.image_encoder = ImageEncoderViT(
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


    # Get the state of the model before loading the new state dict
    # before_state = {name: param.clone() for name, param in model.named_parameters()}

    if model_path:
        # Load the state dict
        model.load_state_dict(torch.load(model_path), strict=False)

    # # Get the state of the model after loading the new state dict
    # after_state = {name: param.clone() for name, param in model.named_parameters()}

    # # Compare the parameters before and after loading the state dict
    # for name, param_before in before_state.items():
    #     param_after = after_state[name]
    #     if not torch.all(param_before.eq(param_after)):
    #         print(f"Parameter {name} was updated.")

    return model

if __name__ == "__main__":


    # 原本的model是这样
    # model = extract_sam_model(image_size=1024)

    # 我把model改成这样
    # model = extract_sam_model(image_size=48)
    # model = model.cuda()

    # 结果有 size mismatch
    # size mismatch for image_encoder.pos_embed: copying a param with shape torch.Size([1, 64, 64, 1280]) from checkpoint, the shape in current model is torch.Size([1, 3, 3, 1280]).
    # size mismatch for image_encoder.blocks.7.attn.rel_pos_h: copying a param with shape torch.Size([127, 80]) from checkpoint, the shape in current model is torch.Size([5, 80]).
    # size mismatch for image_encoder.blocks.7.attn.rel_pos_w: copying a param with shape torch.Size([127, 80]) from checkpoint, the shape in current model is torch.Size([5, 80]).
    # size mismatch for image_encoder.blocks.15.attn.rel_pos_h: copying a param with shape torch.Size([127, 80]) from checkpoint, the shape in current model is torch.Size([5, 80]).
    # size mismatch for image_encoder.blocks.15.attn.rel_pos_w: copying a param with shape torch.Size([127, 80]) from checkpoint, the shape in current model is torch.Size([5, 80]).
    # size mismatch for image_encoder.blocks.23.attn.rel_pos_h: copying a param with shape torch.Size([127, 80]) from checkpoint, the shape in current model is torch.Size([5, 80]).
    # size mismatch for image_encoder.blocks.23.attn.rel_pos_w: copying a param with shape torch.Size([127, 80]) from checkpoint, the shape in current model is torch.Size([5, 80]).
    # size mismatch for image_encoder.blocks.31.attn.rel_pos_h: copying a param with shape torch.Size([127, 80]) from checkpoint, the shape in current model is torch.Size([5, 80]).
    # size mismatch for image_encoder.blocks.31.attn.rel_pos_w: copying a param with shape torch.Size([127, 80]) from checkpoint, the shape in current model is torch.Size([5, 80]).


    # img = np.random.rand(1, 3, 48, 48).astype(np.float32)
    # img = torch.from_numpy(img).cuda()
    # with torch.no_grad():
    #     _, x10, x20, x30 = model.image_encoder(img)
    #     feature = [x10.squeeze(0).cpu().numpy(), x20.squeeze(0).cpu().numpy(), x30.squeeze(0).cpu().numpy()]
    #     x10 = x10.squeeze(0).cpu().numpy()

    #     print(x10.shape)

    model = extract_sam_model(image_size=1024)
    model = model.cuda()

    img = np.zeros((1, 3, 1024, 1024)).astype(np.float32)
    img[:,:,:48,:48] = np.ones((3, 48, 48))

    img = torch.from_numpy(img).cuda()
    with torch.no_grad():
        _, x10, x20, x30 = model.image_encoder(img)
        feature = [x10.squeeze(0).cpu().numpy(), x20.squeeze(0).cpu().numpy(), x30.squeeze(0).cpu().numpy()]
        x10 = x10.squeeze(0).cpu().numpy()

        print(x10.shape)