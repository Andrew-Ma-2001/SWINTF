# 把预处理函数写在外面，不能写在里面了
def check_precompute():
    import numpy as np
    import torch
    import sys
    import os
    sys.path.append('/home/mayanze/PycharmProjects/SwinTF/')
    from PIL import Image
    from utils.utils_data import get_all_images, process_batch
    from data.extract_sam_features import extract_sam_model
    from utils.utils_image import augment_img

    LR_path = '/home/mayanze/PycharmProjects/SwinTF/dataset/trainsets/trainL/DIV2K/DIV2K_train_LR_bicubic/X2'
    LR_size = 48
    pretrained_sam_img_size = 48
    use_cuda = True
    save_path = LR_path + '_yadapt_aug'
    model = extract_sam_model(model_path='/home/mayanze/PycharmProjects/SwinTF/sam_vit_h_4b8939.pth', image_size = 1024)
    # only use 0,1 gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]

    pixel_mean = np.array(pixel_mean)
    pixel_std = np.array(pixel_std)

    if use_cuda:
        model = model.cuda()
        model.image_encoder = torch.nn.DataParallel(model.image_encoder)

    LR_images = get_all_images(LR_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    assert len(os.listdir(save_path)) == 0, "The save_path should be empty"

    modes = [0,1,2,3,4,5,6,7] # 8 modes

    for mode in modes:
        for idx in range(len(LR_images)):
            LR_image = Image.open(LR_images[idx])
            LR_image = np.array(LR_image)
            patches = []

            LR_image = (LR_image - pixel_mean) / pixel_std
            # Cut the LR_image into patches
            for y in range(0, LR_image.shape[0] - LR_size + 1, LR_size):
                for x in range(0, LR_image.shape[1] - LR_size + 1, LR_size):
                    patch = LR_image[y:y + LR_size, x:x + LR_size,:]
                    patch = augment_img(patch, mode)
                    patches.append((patch, x, y))

            batch_LR_image = np.zeros((len(patches), 3, pretrained_sam_img_size, pretrained_sam_img_size))
            for i, (patch, _, _) in enumerate(patches):
                batch_LR_image[i] = patch.transpose(2, 0, 1)

            # 这里要把 48x48 变成 1024x1024 建一个更大的矩阵
            large_img = np.zeros((batch_LR_image.shape[0], 3, 1024, 1024))
            large_img[:, :, :48, :48] = batch_LR_image
            # 然后将 batch_LR_image 转换成 tensor
            batch_LR_image_sam = torch.from_numpy(large_img).float()
            # 然后将 batch_LR_image 输入到模型中
            inferece_batch_size = 15

            if batch_LR_image_sam.shape[0] <= inferece_batch_size:
                if use_cuda:
                    batch_LR_image_sam = batch_LR_image_sam.cuda()

                with torch.no_grad():
                    _, y1, y2, y3 = model.image_encoder(batch_LR_image_sam)
                    y1, y2, y3 = y1.cpu().numpy(), y2.cpu().numpy(), y3.cpu().numpy()
            else:
                if use_cuda:
                    y1, y2, y3 = process_batch(batch_LR_image_sam, model.image_encoder, inferece_batch_size)

            y1, y2, y3 = y1[:, :, :3, :3], y2[:, :, :3, :3], y3[:, :, :3, :3]
            yadapt_features = np.concatenate((y1, y2, y3), axis=1)

            # 对于 yadapt_features 分别保存
            for i in range(yadapt_features.shape[0]):
                save_name = os.path.join(save_path, os.path.basename(LR_images[idx]).split(".")[0]+'_'+str(i)+'_'+str(mode)+'_yadapt.npy')
                np.save(save_name, yadapt_features[i])
                print('Save {}'.format(save_name))

check_precompute()