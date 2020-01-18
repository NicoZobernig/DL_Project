from zsldataset import ZSLDataset

# %%



# %%

import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from irevnet import iRevNet, iRevNet_ZSL

# %%

model = iRevNet_ZSL(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                    nChannels=[24, 96, 384, 1536], nClasses=1000, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[3, 224, 224])

model = model.cuda()
model = model.eval()

d = torch.load('irevnet-pretrained.pth.tar')

state_dict = d['state_dict']

from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)





def pad_and_resize(img, desired_size):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(img, ((desired_size - new_size[0]) // 2,
                       (desired_size - new_size[1]) // 2))

    return new_im


dataset_path='Data/CUB/'
has_bounding_box = True
image_path ='Data/CUB_200_2011/CUB_200_2011/images/'
filenames = pd.read_csv(dataset_path+'/filenames_labels.txt', index_col=0, header=0)
embeddings = []

if has_bounding_box:
    bounding_boxes = pd.read_csv(dataset_path+'bounding_boxes.txt',delimiter=' ', index_col=0, header=None)


for i in range(0, len(filenames)):
    img = Image.open(image_path + filenames.iloc[i][0])

    if has_bounding_box:
        #Convert bounding box from (left,upper,width,heigth) to (left, upper, right, lower)
        left = bounding_boxes.iloc[i][1]
        upper = bounding_boxes.iloc[i][2]
        right = left + bounding_boxes.iloc[i][3]
        lower = upper + bounding_boxes.iloc[i][4]
        img = img.crop((left, upper, right, lower))

    img = pad_and_resize(img, 224)

    with torch.no_grad():
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        img_in = normalize(to_tensor(img)).unsqueeze(0).cuda()

        features = model.forward_features(img_in)

        embeddings.append(features.cpu().data.numpy().squeeze().tolist())




indx = range(0, len(filenames))
out_df = pd.DataFrame(embeddings, columns=range(0, len(embeddings[0])))
out_df.insert(0, 'sample_id', indx)
out_df.to_csv('Data/CUB/irevnet_image_embeddings_boxed.txt', index=0)
