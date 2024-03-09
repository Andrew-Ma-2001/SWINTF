from data.extract_sam_features import extract_sam_model
import torch
import numpy as np
import PIL.Image as Image

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
