import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from nets.swinir import SwinIR as net
from nets.swinir import SwinIRAdapter as net_adapter
from utils import util_calculate_psnr_ssim as util

import yaml
from data.extract_sam_features import extract_sam_model
from compute_dataset import ImagePreprocessor
from utils.utils_image import imresize_np







def main():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    scale = config['train']['scale']
    training_patch_size = config['train']['patch_size'] // scale
    # tile = training_patch_size
    tile = None
    tile_overlap = 32

    config['test']['tile'] = tile
    config['test']['tile_overlap'] = tile_overlap
    config['network']['swinir_test'] = test_swinir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        raise ValueError(f'model path {model_path} does not exist')

    model = define_model(scale, training_patch_size, model_path, config)
    model.eval()
    model = model.to(device)

    if not test_swinir:
        sam_model, preprocessor = define_sam_model(model_path, training_patch_size)

    # setup folder and path
    folder, save_dir, border, window_size = setup(config)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []
    psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        if test_swinir:
            imgname, img_lq, img_gt = get_image_pair(config, path)  # image to HWC-BGR, float32
        else:
            imgname, img_lq, img_gt, y_adapt_features = get_image_pair(config, path, sam_model=sam_model, preprocessor=preprocessor)  # image to HWC-BGR, float32

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            if test_swinir:
                output = test(img_lq, model, config, window_size)
            else:
                output = test(img_lq, model, config, window_size, y_adapt_features=y_adapt_features)

            output = output[..., :h_old * config["test"]["scale"], :w_old * config["test"]["scale"]]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

        cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)
        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * config["test"]["scale"], :w_old * config["test"]["scale"], ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            try:
                assert output.shape == img_gt.shape
            except AssertionError:
                # Force the output to be the same shape as img_gt
                print(f"Output shape: {output.shape}, GT shape: {img_gt.shape}")
                output = output[:img_gt.shape[0], :img_gt.shape[1], :]
                
            psnr = util.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util.calculate_ssim(output, img_gt, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNRB: {:.2f} dB;'
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; PSNRB_Y: {:.2f} dB.'.
                  format(idx, imgname, psnr, ssim, psnrb, psnr_y, ssim_y, psnrb_y))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))


def define_model(scale, patch_size, model_path, config):
    if config['network']['swinir_test'] is True:
        model = net(upscale=scale, in_chans=3, img_size=patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        return model
    else:   
        model = net_adapter(upscale=scale, in_chans=3, img_size=patch_size, window_size=8,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv', y_adapt_feature=torch.zeros(1, 1, 1, 1))
        
        model = torch.nn.DataParallel(model)
        pretrained_model = torch.load(model_path)
        # param_key_g = 'params'
        # model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=False)
        model.load_state_dict(pretrained_model, strict=True)
        model.cuda()
        # model = torch.nn.DataParallel(model)
        return model
    

def define_sam_model(model_path, image_size):
    model = extract_sam_model(model_path='/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth', image_size = 1024)
    model = model.cuda()
    model.image_encoder = torch.nn.DataParallel(model.image_encoder)
    preprocessor = ImagePreprocessor()

    return model, preprocessor

def setup(config):
    # 001 classical image sr/ 002 lightweight image sr
    noise_setting = config['test']['test_LR'].split('/')[-1]
    if test_swinir:
        save_dir = f'results/swinir_classical_sr_x{config["test"]["scale"]}_{noise_setting}'
    else:
        save_dir = f'results/swinir_adapter_sr_x{config["test"]["scale"]}_{noise_setting}'
    folder = config["test"]["test_HR"]
    border = config["test"]["scale"]
    window_size = 8 * 2 

    return folder, save_dir, border, window_size


def get_image_pair(config, path, **kwargs):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{config["test"]["test_LR"]}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.
    
    if config['network']['swinir_test'] is True:
        return imgname, img_lq, img_gt
    else:
        # Calculate the yadapt features
        sam_model = kwargs.get('sam_model', None)
        preprocessor = kwargs.get('preprocessor', None)
        y_adapt_features = calculate_yadapt_features(img_lq, sam_model, preprocessor)
        # Change y_adapt_features to tensor
        y_adapt_features = torch.tensor(y_adapt_features).float()
        y_adapt_features = y_adapt_features.unsqueeze(0)
        return imgname, img_lq, img_gt, y_adapt_features

def calculate_yadapt_features(img_lq, sam_model, preprocessor):
    preprocessor.set_image(img_lq)
    torch_img = preprocessor.preprocess_image_v2(device='cuda')
    with torch.no_grad():
        _, y1, y2, y3 = sam_model.image_encoder(torch_img)
    # Concate y1 y2 y3 by torch
    y = torch.cat([y1, y2, y3], dim=1)
    y = y.cpu().numpy()
    y = preprocessor.slice_yadapt_features(y)
    return y

def test(img_lq, model, config, window_size, **kwargs):
    y_adapt_features = kwargs.get('y_adapt_features', None)
    if config["test"]["tile"] is None:
        with torch.no_grad():
            if config['network']['swinir_test'] is True:
                # test the image as a whole
                output = model(img_lq)
            else:
                # test the image as a whole
                output = model(x=img_lq, y_adapt_feature=y_adapt_features)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(config["test"]["tile"], h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = config["test"]["tile_overlap"]
        sf = config["test"]["scale"]

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)


        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                if config['network']['swinir_test'] is True:
                    out_patch = model(in_patch)
                else:
                    # out_patch = model(in_patch, y_adapt_feature=y_adapt_features[..., ])
                    raise NotImplementedError("Y-adapt feature is not implemented for tile-by-tile testing")
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


def test_main(config_path, model, test_swinir, save_img=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    scale = config['train']['scale']
    training_patch_size = config['train']['patch_size'] // scale
    tile = None
    tile_overlap = 32

    config['test']['tile'] = tile
    config['test']['tile_overlap'] = tile_overlap
    config['network']['swinir_test'] = test_swinir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if not test_swinir:
        sam_model, preprocessor = define_sam_model(model_path=None, image_size=48)

    folder, save_dir, border, window_size = setup(config)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        if test_swinir:
            imgname, img_lq, img_gt = get_image_pair(config, path)
        else:
            imgname, img_lq, img_gt, y_adapt_features = get_image_pair(config, path, sam_model=sam_model, preprocessor=preprocessor)

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            if test_swinir:
                output = test(img_lq, model, config, window_size)
            else:
                output = test(img_lq, model, config, window_size, y_adapt_features=y_adapt_features)

            output = output[..., :h_old * config["test"]["scale"], :w_old * config["test"]["scale"]]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)
            img_gt = img_gt[:h_old * config["test"]["scale"], :w_old * config["test"]["scale"], ...]
            img_gt = np.squeeze(img_gt)

            try:
                assert output.shape == img_gt.shape
                if save_img:
                    cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)
            except AssertionError:
                print(f"Output shape: {output.shape}, GT shape: {img_gt.shape}")
                output = output[:img_gt.shape[0], :img_gt.shape[1], :]

            psnr = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
            print(f"PSNR: {psnr}")
            test_results['psnr'].append(psnr)

    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        return ave_psnr


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

    def parse_args():
        parser = argparse.ArgumentParser(description='Test SwinIR model')
        parser.add_argument('--config', type=str, required=True, help='Path to the config file')
        parser.add_argument('--model', type=str, required=True, help='Path to the model file')
        parser.add_argument('--test_swinir', action='store_true', help='Whether to test SwinIR model')
        parser.add_argument('--gpu', type=str, default='0', help='GPU id(s) to use (comma-separated, e.g., "0,1,2,3")')
        parser.add_argument('--save_img', action='store_true', help='Whether to save the output image')
        return parser.parse_args()

    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config_path = args.config
    model_path = args.model
    test_swinir = args.test_swinir
    save_img = args.save_img
    import time
    start_time = time.time()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    scale = config['train']['scale']
    training_patch_size = config['train']['patch_size'] // scale
    # tile = training_patch_size
    tile = None
    tile_overlap = 32
    config['test']['tile'] = tile
    config['test']['tile_overlap'] = tile_overlap
    config['network']['swinir_test'] = test_swinir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # set up model
    if os.path.exists(model_path):
        print(f'loading model from {model_path}')
    else:
        raise ValueError(f'model path {model_path} does not exist')
    main()

    # For testing test_main function
    # model = define_model(scale, training_patch_size, model_path, config)
    # model.eval()
    # model = model.to(device)

    # print(test_main(config_path, model, test_swinir))
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")

    # test_main(config_path, model, test_swinir)

    # del model
    # print("Testing SwinIR model")
    # model = define_model(scale, training_patch_size, model_path, config)
    # model.eval()
    # model = model.to(device)
    # test_main(config_path, model, test_swinir)

    # /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240813062655/50000_model.pth