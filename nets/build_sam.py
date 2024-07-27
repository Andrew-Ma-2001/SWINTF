import torch

from functools import partial

import sys
sys.path.append("/home/mayanze/PycharmProjects/SwinTF/")

from nets.image_encoder_vit import ImageEncoderViT, ImageEncoderViT_FeatureExtract
from nets.mask_decoder import MaskDecoder
from nets.prompt_encoder import PromptEncoder
from nets.sam import Sam
from nets.transformer import TwoWayTransformer


# from image_encoder_vit import ImageEncoderViT, ImageEncoderViT_FeatureExtract
# from mask_decoder import MaskDecoder
# from prompt_encoder import PromptEncoder
# from sam import Sam
# from transformer import TwoWayTransformer


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
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
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

def build_sam_vit_l(**kwargs):
    pass

def build_sam_vit_b(**kwargs):
    pass

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

if __name__ == "__main__":
    # Using ViT-h
    encoder_embed_dim=1280
    encoder_depth=32
    encoder_num_heads=16
    encoder_global_attn_indexes=[7, 15, 23, 31]
    # checkpoint=checkpoint
    prompt_embed_dim = 256

    # test_image [200, 200, 3]
    # 
    image_size = 500
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
    # Create Fake Image
    fake_image = torch.randn(1, 3, 500, 500)
    output = image_encoder(fake_image)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)

    model_path = '/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth'
    # image_encoder.load_state_dict(torch.load(model_path), strict=False)

    # Assume `model` is your model and `partial_state_dict` is the state dict you want to load
    model = image_encoder
    # Get the state of the model before loading the new state dict
    before_state = {name: param.clone() for name, param in model.named_parameters()}

    # Load the state dict
    model.load_state_dict(torch.load(model_path), strict=False)

    # Get the state of the model after loading the new state dict
    after_state = {name: param.clone() for name, param in model.named_parameters()}

    # Compare the parameters before and after loading the state dict
    for name, param_before in before_state.items():
        param_after = after_state[name]
        if not torch.all(param_before.eq(param_after)):
            print(f"Parameter {name} was updated.")

    output = image_encoder(fake_image)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)


    # sam_model = build_sam_vit_h(model_path)
    # fake_image = torch.randn(48, 48, 3)
    # output = sam_model(fake_image, multimask_output=False)

    a = 1

    import torch 
    image = torch.randn(1, 3, 480, 480)
    
    image2 = image.view(1, 3, 10, 48, 10, 48)
    
    image3 = image2.permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, 48, 48)
    print(image3.shape)

    # 100xCx48x48 -> SwinIR -> 1xCx480x480
    # SwinIRAdapt (x, y) x 1xC*480*480 , y 100xCx3x3
    # SwinIRAdapt (x, y) x 1xC*480*480 , 
    
    # Reshape 回去



