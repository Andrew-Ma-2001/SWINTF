import torch
import torch.nn as nn
from nets.swinir import SwinIR
from nets.build_sam import build_sam_vit_h, _build_sam


class SwinIR_SAM(nn.Module):
    def __init__(self, upscale=2, in_chans=3, img_size=(64, 64), window_size=8, img_range=1., 
                 depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                 upsampler='pixelshuffledirect', resi_connection='1conv', 
                 y_adapt_feature=None, checkpoint_sam='/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth'):
        super().__init__()
        self.sam = build_sam_vit_h(checkpoint_sam)
        self.proj = nn.Conv2d(1280, embed_dim*16, 1, 1, 0)

        self.swinir = SwinIR(
            upscale=upscale, img_size=img_size,
            window_size=window_size, img_range=img_range, depths=depths,
            embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, upsampler=upsampler,
            resi_connection=resi_connection
        )

    def forward(self, img, y_adapt=None):
        feats_sam_list = self.sam.image_encoder(img)
        B, H, W, C = feats_sam_list[0].shape
        feats_adapt_list = []
        for feats_sam in feats_sam_list:
            feats_adapt = self.proj(feats_sam.permute(0, 3, 1, 2))
            feats_adapt_list.append(feats_adapt)
        out = self.swinir(img, feats_adapt_list)

        return out