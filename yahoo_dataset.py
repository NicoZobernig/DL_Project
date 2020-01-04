import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class YahooDataset(Dataset):
    """ Yahoo images with attributes """

    def __init__(self, dataset_path, test, auto_crop = False):
        self.path = dataset_path
        
        self.yahoo_path = os.path.join(dataset_path, "ayahoo_test_images") 
        self.pascal_path = os.path.join(dataset_path, "VOCdevkit/VOC2008/JPEGImages")
        self.attributes_path = os.path.join(dataset_path, "attribute_data")
        
        self.test = test

        # load class names
        with open(os.path.join(self.attributes_path, 'class_names.txt')) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [x for x in content if x != ""]
        
        self.class_names = content
        
        # load class names
        with open(os.path.join(self.attributes_path, 'attribute_names.txt')) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [x for x in content if x != ""]
        
        self.attribute_names = content 
        
        self.load_attributes()
        
        self.auto_crop = True
        
        
    def load_attributes(self):
        
        if self.test:
            # load yahoo test images
            with open(os.path.join(self.attributes_path, 'ayahoo_test.txt')) as f:
                content = f.readlines()
            content = [x.strip().split() for x in content]
            
            self.data = [(os.path.join(self.yahoo_path, x[0]), 
                          x[1], 
                          np.array(x[2:6], dtype=np.float32), 
                          np.array(x[6:], dtype=np.float32)) 
                             for x in content]
            
            # load yahoo test images
            with open(os.path.join(self.attributes_path, 'apascal_test.txt')) as f:
                content = f.readlines()
            content = [x.strip().split() for x in content]
            
            self.data = self.data + [(os.path.join(self.pascal_path, x[0]), 
                                      x[1], 
                                      np.array(x[2:6], dtype=np.float32),
                                      np.array(x[6:], dtype=np.float32)) 
                                         for x in content]
            
            
        else:
            # load yahoo test images
            with open(os.path.join(self.attributes_path, 'apascal_train.txt')) as f:
                content = f.readlines()
            content = [x.strip().split() for x in content]
            
            self.data = [(os.path.join(self.pascal_path, x[0]), 
                          x[1], 
                          np.array(x[2:6], dtype=np.float32), 
                          np.array(x[6:], dtype=np.float32)) 
                             for x in content]
            
            
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, class_name, bbox, attributes = self.data[idx]
        image = Image.open(img_file)
        
        if self.auto_crop:
            image = image.crop(bbox)
            
            return (image, class_name, attributes)
        else:
            return (image, class_name, bbox, attributes)


