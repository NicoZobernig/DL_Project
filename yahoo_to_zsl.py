import argparse
import os
import pandas as pd
from yahoo_dataset import YahooDataset
import re
import numpy as np
from PIL import Image
import torch
from irevnet import iRevNet, iRevNet_ZSL
from torchvision import transforms


parser = argparse.ArgumentParser()
   
parser.add_argument('-y', '--yahoo', required=True, help="Yahoo dir")
parser.add_argument('-o', '--output', required=True, help="Output dir")

args = parser.parse_args()


def pad_and_resize(img, desired_size):

    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    img = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_im


if not os.path.exists(args.output):
    os.mkdir(args.output)
    
testpath = os.path.join(args.output, 'test')
trainpath = os.path.join(args.output, 'train')


if not os.path.exists(testpath):
    os.mkdir(testpath)
    
if not os.path.exists(trainpath):
    os.mkdir(trainpath)
    
trainset = YahooDataset(args.yahoo, test = False, auto_crop = True)
testset = YahooDataset(args.yahoo, test = True, auto_crop = True)

print('collecting class names...')

class_names = trainset.class_names

with open(os.path.join(trainpath, 'classes.txt'), 'w') as f:
    f.write('class_id,label')
    for i,c in enumerate(class_names):
        f.write("\n%i,%s" % (i, c))

with open(os.path.join(testpath, 'classes.txt'), 'w') as f:
    f.write('class_id,label')
    for i,c in enumerate(class_names):
        f.write("\n%i,%s" % (i, c))


def pad_and_resize(img, desired_size):

    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    img = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_im


print("generating glove embeddings...")

word_embeddings = []

map_to_glove = {'diningtable':'table', 'tvmonitor':'monitor', 'pottedplant':'plant'}

for cname in class_names:
    if cname in map_to_glove:
        cname = map_to_glove[cname]
    with open('./glove/glove.6B.300d.txt') as reader:
        found = False
        
        for line in reader:
            if re.match(r'{0} '.format(cname), line):
                line_split = line.rstrip().split(' ')
#                 found_labels.append(line_split[0])
                word_embeddings.append(np.asarray(line_split[1:], dtype=float))
                found = True
                break  
        if not found:
            print('Class not found: {}'.format(cname))
            assert found
            
out_df = pd.DataFrame(word_embeddings, columns=range(0,300))
out_df.insert(0, 'class_name', class_names)
out_df.to_csv(os.path.join(testpath, 'glove_embeddings_300.txt'), sep=',', index=False)
out_df.to_csv(os.path.join(trainpath, 'glove_embeddings_300.txt'), sep=',', index=False)


####

print("generating visual embeddings...")

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
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
    
model.load_state_dict(new_state_dict)

print('- for train set')

embeddings = []

for i in range(len(trainset)):
    d = trainset[i]
    
    img = d[0]
    img = pad_and_resize(img, 224)
    
    with torch.no_grad():
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        img_in = normalize(to_tensor(img)).unsqueeze(0).cuda()

        features = model.forward_features(img_in)
        
        embeddings.append(features.cpu().data.numpy().squeeze())
        
    if i % 10 == 0:    
        print('.', end='')
        
out_df = pd.DataFrame(embeddings, columns=range(embeddings[0].shape[0]))
out_df.insert(0, 'sample_id', range(len(embeddings)))
out_df.to_csv(os.path.join(trainpath, 'irevnet_image_embeddings.txt'), sep=',', index=False)
        
del embeddings, out_df


print('- for test set')

embeddings = []

# there seems to be an issue with entry 1525 in the test set
a,b,c,d = testset.data[1525]
testset.data[1525] = (a,b,np.array([31., 22., 314., 314.]),d)

a,b,c,d = testset.data[2210]
testset.data[2210] = (a,b,np.array([0., 100., 333., 500.]),d)

for i in range(len(testset)):
    d = testset[i]
    
    img = d[0]
    img = pad_and_resize(img, 224)
    
    with torch.no_grad():
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        img_in = normalize(to_tensor(img)).unsqueeze(0).cuda()

        features = model.forward_features(img_in)
        
        embeddings.append(features.cpu().data.numpy().squeeze())
        
    if i % 10 == 0:    
        print('.', end='')
        
out_df = pd.DataFrame(embeddings, columns=range(embeddings[0].shape[0]))
out_df.insert(0, 'sample_id', range(len(embeddings)))
out_df.to_csv(os.path.join(testpath, 'irevnet_image_embeddings.txt'), sep=',', index=False)
        
del embeddings, out_df


#####
print('generating indices...')

print('- for train set')

file_names = [e[0][len(trainset.path):] for e in trainset.data]
class_ids = [class_names.index(e[1]) for e in trainset.data]

out_df = pd.DataFrame(class_ids, columns=['class_id'])
out_df.insert(0, 'filename', file_names)
out_df.insert(0, 'sample_id', range(len(file_names)))

out_df.to_csv(os.path.join(trainpath, 'filenames_labels.txt'), sep=',', index=False)

print('- for test set')

file_names = [e[0][len(testset.path):] for e in testset.data]
class_ids = [class_names.index(e[1]) for e in testset.data]

out_df = pd.DataFrame(class_ids, columns=['class_id'])
out_df.insert(0, 'filename', file_names)
out_df.insert(0, 'sample_id', range(len(file_names)))

out_df.to_csv(os.path.join(testpath, 'filenames_labels.txt'), sep=',', index=False)


