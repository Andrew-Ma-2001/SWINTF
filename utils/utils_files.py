import numpy as np
from data.extract_sam_features import extract_sam_model
import torch
import numpy as np
import PIL.Image as Image


def check_npy_equal(npy1, npy2):
    data1 = np.load(npy1)
    data2 = np.load(npy2)
    if data1.shape != data2.shape:
        print(f"Shape of {npy1} is {data1.shape}, and shape of {npy2} is {data2.shape}.")
        return False
    if not np.allclose(data1, data2):
        print(f"{npy1} and {npy2} are not equal.")
        return False
    return True


def check_model_randomness():

    # 1 Load in the image
    example_img = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set5/LRbicx2/baby.png'
    img = np.array(Image.open(example_img))[:48, :48, :]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()

    # Create a 1024x1024 image randomly
    img = torch.rand(1, 3, 48, 48).cuda()



    model = extract_sam_model(image_size=1024)
    model = model.cuda()

    # 2 Run the inference two times and compare the results
    with torch.no_grad():
        # Get the feature
        _, x10, x20, x30 = model.image_encoder(img)
        feature = [x10.squeeze(0).cpu().numpy(), x20.squeeze(0).cpu().numpy(), x30.squeeze(0).cpu().numpy()]

        # Make feature a numpy array 
        feature = np.array(feature)

    model2 = extract_sam_model(image_size=1024)
    model2 = model2.cuda()

    with torch.no_grad():
        # Get the feature
        _, x10, x20, x30 = model2.image_encoder(img)
        feature2 = [x10.squeeze(0).cpu().numpy(), x20.squeeze(0).cpu().numpy(), x30.squeeze(0).cpu().numpy()]

        # Make feature a numpy array 
        feature2 = np.array(feature2)

    # 3 Compare the results
    print(np.allclose(feature, feature2))


def extract_sam_features():
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
    #     # Using 这里假如用了 Resize 原则上在数据读入的时候就需要？Dataloader？
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
    #     # Using 这里假如用了 Resize 原则上在数据读入的时候就需要？Dataloader？
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
    pass

if __name__ == "__main__":
    # Now this code is checking filling zeros
    model = extract_sam_model(image_size=1024)
    model = model.cuda()

    example_img = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set5/LRbicx2/baby.png'
    img = np.array(Image.open(example_img))[:48, :48, :]

    # Create a 1024x1024x3 matrix
    img_large = np.zeros((1024, 1024, 3), dtype=np.uint8)
    img_large[:48, :48, :] = img

    img_large_tensor = torch.from_numpy(img_large).permute(2, 0, 1).unsqueeze(0).float().cuda()
    with torch.no_grad():
        _, x10, x20, x30 = model.image_encoder(img_large_tensor)
        # Only take out 3x3 in dimension 2,3
        x10 = x10.squeeze(0).cpu().numpy()[:, :3, :3]
        x20 = x20.squeeze(0).cpu().numpy()[:, :3, :3]
        x30 = x30.squeeze(0).cpu().numpy()[:, :3, :3]

        print(x10.shape)
