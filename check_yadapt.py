from data.extract_sam_features import extract_sam_model
import torch
import numpy as np
import PIL.Image as Image


# 1 Load in the image
example_img = '/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set5/LRbicx2/baby.png'
img = np.array(Image.open(example_img))[:48, :48, :]
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()


model = extract_sam_model()
model = model.cuda()

# 2 Run the inference two times and compare the results
with torch.no_grad():
    # Get the feature
    _, x10, x20, x30 = model(img)
    feature = [x10.squeeze(0).cpu().numpy(), x20.squeeze(0).cpu().numpy(), x30.squeeze(0).cpu().numpy()]

    # Make feature a numpy array 
    feature = np.array(feature)

model2 = extract_sam_model()
model2 = model2.cuda()

with torch.no_grad():
    # Get the feature
    _, x10, x20, x30 = model2(img)
    feature2 = [x10.squeeze(0).cpu().numpy(), x20.squeeze(0).cpu().numpy(), x30.squeeze(0).cpu().numpy()]

    # Make feature a numpy array 
    feature2 = np.array(feature2)

# 3 Compare the results
print(np.allclose(feature, feature2))

