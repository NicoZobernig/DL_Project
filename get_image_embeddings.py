import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_image_embedding(image_name=''):
    img = cv2.imread(image_name)
    img = img.transpose((2, 0, 1))/np.amax(img)  # transform into correct shape and scale to 0-1

    torch_img = torch.Tensor(img)
    torch_img = normalize(torch_img)
    var_img = Variable(torch_img)

    # Load the pre-trained model and use the architecture as new nn
    model = models.resnet101(pretrained=True)
    modules = list(model.children())[:-1]  # keep network until last layer (fc)
    model = nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False

    features_var = model(var_img.unsqueeze(0))
    features = features_var.data

    return features.squeeze()
