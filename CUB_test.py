from Datasets import CUBDataset
from utils import get_word_embedding_glove
from utils import get_CUB_word_embedding_glove
import torch
import pandas as pd
import cv2


# Test GloVe embedding function

glove_path_300 = 'Data/glove/glove.6B.300d.txt'

label = 'catbird'
composite_label = 'gray+catbird'


found_labels_300, word_embeddings_300 = get_word_embedding_glove(glove_path_300, composite_label, alpha=0.7)
torch_embedding_300 = torch.from_numpy(word_embeddings_300)
print(found_labels_300)
print(torch_embedding_300.shape)





# Test AwA2 Dataset structure
test_with = CUBDataset(dataset_path='Data/CUB_200_2011/CUB_200_2011/',
                       class_embedding_path='Data/CUB_200_2011/CUB_200_2011/CUB_glove_embedding.txt',
                       image_path='Data/CUB/images')

sample_with = test_with[0]
print('Sample: ', sample_with.keys())
test_with.show_info()
#
# test_without = AwA2Dataset(dataset_path='Data/AwA2/',
#                            class_embedding_path='Data/AwA2/glove_embeddings_300.txt',
#                            image_path='Data/AwA2/img',
#                            use_predicates=False)
#
# sample_without = test_without[0]
# print('Sample: ', sample_without.keys())
# test_without.show_info()

# cv2.imshow('test', sample_with['image'])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

