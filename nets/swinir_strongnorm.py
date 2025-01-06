# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from nets.swinir import RSTB, PatchEmbed, PatchUnEmbed, Upsample, UpsampleOneStep

class MinMaxNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_val = None
        self.max_val = None

    def forward(self, x, reverse=False):
        if not reverse:
            # Store parameters and normalize to [-1, 1]
            self.min_val = x.min()
            self.max_val = x.max()
            x_normalized = 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1
            return x_normalized
        else:
            # Denormalize back to original range
            if self.min_val is None or self.max_val is None:
                raise ValueError("Cannot reverse normalize without first normalizing")
            x_denormalized = (x + 1) * (self.max_val - self.min_val) / 2 + self.min_val
            return x_denormalized
        
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.q = nn.Linear(in_channels, in_channels)
        self.kv = nn.Linear(in_channels, in_channels*2)
        self.proj = nn.Linear(in_channels, in_channels)
        self.tau = nn.Parameter(torch.zeros(1))
        self.minmax_norm = MinMaxNorm()

    def forward(self, y_adapt, x):
        '''
        y_adapt: B, C, M, N
        x: B, C, H, W
        '''
        ################### 2024-03-23 ##########################
        batch_size, C, height, width = x.size()
        y_adapt_flatten = y_adapt.flatten(2).transpose(1, 2)    # B, MN, C
        x_flatten = x.flatten(2).transpose(1, 2)    # B, HW, C

       ################### 2024-04-16 ##########################
        # q = self.q(self.norm1(x_flatten))  # B, HW, C
        # kv = self.kv(self.norm2(y_adapt_flatten)).view(batch_size, -1, 2, C)

       #################### 2024-11-30 不用 norm ##########################
        x_flatten = self.minmax_norm(x_flatten)
        y_adapt_flatten = self.minmax_norm(y_adapt_flatten)

        q = self.q(x_flatten)  # B, HW, C
        kv = self.kv(y_adapt_flatten).view(batch_size, -1, 2, C)
        k, v = kv.unbind(2)  # B, HW, C

        attn = (q @ k.transpose(-2, -1)) / C #是C还是根号C看经验
        attn = attn.softmax(dim=-1)
        # breakpoint()
        out = (attn @ v)  # bug here

       ################### 2024-08-07 ##########################
        # mode 1: static hyperparameter
        out = self.minmax_norm(out, reverse=True)
        out = self.tau * self.proj(out).transpose(-1, -2).reshape(batch_size, -1, height, width) + x

        return out


class SwinIRStrongNorm(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv', y_adapt_feature=None,
                 **kwargs):
        super(SwinIRStrongNorm, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # i_layer in range 6 
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # XXX 这里把 self_attention 的大小写死了
        self.y_adapt_feature_size = y_adapt_feature
        self.yadapt_batch_norm = nn.BatchNorm2d(180)

        self.adapt_conv = nn.ModuleList([
            nn.Conv2d(3840, 180, kernel_size=1),
            nn.Conv2d(3840, 180, kernel_size=1),
            nn.Conv2d(3840, 180, kernel_size=1),
            nn.Conv2d(3840, 180, kernel_size=1),
            nn.Conv2d(3840, 180, kernel_size=1),
            nn.Conv2d(3840, 180, kernel_size=1),
        ])

        self.adapt_self_attention = nn.ModuleList([
            SelfAttention(180),
            SelfAttention(180),
            SelfAttention(180),
            SelfAttention(180),
            SelfAttention(180),
            SelfAttention(180),
        ])

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x, y_adapt_feature):
        '''
        y_adapt_feature: B, 3840, 3, 3
        '''
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x, x_size) # torch.Size([1, 4096, 256])

            # ===========================================================================
            # ========================  加入 adapter 机制  ===============================
            # ===========================================================================
            if y_adapt_feature is not None:
                x = self.patch_unembed(x, x_size)  # torch.Size([1, 180, 48, 48])

                # Apply ReLU after the convolution, not before
                # y_adapt = self.adapt_conv_first[idx](y_adapt_feature)
                # y_adapt = F.relu(y_adapt)  # Apply ReLU after conv
                # y_adapt = self.adapt_conv_last[idx](y_adapt)
                # x = self.x_conv_first[idx](x)
                # x = self.self_attention[idx](y_adapt, x)
                # x = self.x_conv_last[idx](x)

                y_adapt = self.adapt_conv[idx](y_adapt_feature)
                x = self.adapt_self_attention[idx](y_adapt, x)

                # y_adapt = y_adapt.view(-1, 180, 48, 48)
                # y_adapt = self.yadapt_batch_norm(y_adapt)
                # Reshape 回去
                # y_adapt1 = y_adapt.view(1, 10, 10, 180, 48, 48)
                # y_adapt2 = y_adapt1.permute(0, 3, 1, 4, 2, 5).contiguous().view(1, 180, 10*48, 10*48)

                ################ 2024-03-23 ####################
                ## x = self.patch_embed(x)
                x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C Avoiding having batch norm
                ## ===========================================================================
                # idx += 1

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, y_adapt_feature):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, y_adapt_feature)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, y_adapt_feature)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, y_adapt_feature)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, y_adapt_feature)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops




if __name__ == '__main__':
    # upscale = 4
    # window_size = 8
    # height = (1024 // upscale // window_size + 1) * window_size
    # width = (720 // upscale // window_size + 1) * window_size
    # model = SwinIR(upscale=2, img_size=(height, width),
    #                window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
    #                embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    # print(model)
    # print(height, width, model.flops() / 1e9)

    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size) * 2
    width = (1024 // upscale // window_size) * 2

    height = 48
    width = 48

    # model = SwinIR(upscale=4, img_size=(height, width),
    #                window_size=window_size, img_range=1., depths=[6,6,6,6,6,6],
    #                embed_dim=180, num_heads=[6,6,6,6,6,6], mlp_ratio=2, upsampler='pixelshuffledirect', y_adapt_feature=torch.randn(1, 3480, 3, 3))

    # x = torch.randn((1, 3, height, width))
    # x = model(x)
    # print(x.shape)
    # print(height, width)

    # adapt_conv = nn.Conv2d(3840, 180, kernel_size=1)
    # attention_model = SelfAttention(180)
    # y_adapt = torch.randn((1, 3840, 3, 3))
    # y_adapt = adapt_conv(y_adapt)
    # # y_adapt_conv = nn.Conv2d(180, 1024, kernel_size=1)
    # x = torch.randn((1, 180, 64, 64))
    # x = attention_model(y_adapt, x)
    # print(x.shape)

    model = SwinIRStrongNorm(upscale=2, img_size=(48, 48), window_size=8, 
                          img_range=1., depths=[6,6,6,6,6,6], embed_dim=180, 
                          num_heads=[6,6,6,6,6,6], mlp_ratio=2, upsampler='pixelshuffledirect')
    
    input_test_image = torch.randn(1, 3, 48, 48)
    y_adapt_feature = torch.randn(1, 3840, 3, 3)

    output = model(input_test_image, y_adapt_feature)