from Datasets import AwA2Dataset
from utils import get_word_embedding_glove
import torch
import cv2


# Test GloVe embedding function
glove_path_50 = 'Data/glove/glove.6B.50d.txt'
glove_path_300 = 'Data/glove/glove.6B.300d.txt'

label = 'antelope'
composite_label = 'blue+whale'

found_labels_50, word_embeddings_50 = get_word_embedding_glove(glove_path_50, label)
torch_embedding_50 = torch.from_numpy(word_embeddings_50)
print(found_labels_50)
print(torch_embedding_50.shape)

found_labels_300, word_embeddings_300 = get_word_embedding_glove(glove_path_300, composite_label, alpha=0.7)
torch_embedding_300 = torch.from_numpy(word_embeddings_300)
print(found_labels_300)
print(torch_embedding_300.shape)


# Test AwA2 Dataset structure
test_with = AwA2Dataset(dataset_path='Data/AwA2/',
                        class_embedding_path='Data/AwA2/glove_embeddings_300.txt',
                        image_path='Data/AwA2/img')

sample_with = test_with[0]
print('Sample: ', sample_with.keys())
test_with.show_info()

test_without = AwA2Dataset(dataset_path='Data/AwA2/',
                           class_embedding_path='Data/AwA2/glove_embeddings_300.txt',
                           image_path='Data/AwA2/img',
                           use_predicates=False)

sample_without = test_without[0]
print('Sample: ', sample_without.keys())
test_without.show_info()

# cv2.imshow('test', sample_with['image'])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

