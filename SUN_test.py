from Datasets import SUNDataset
#from utils import get_word_embedding_glove
from utils import get_SUN_word_embedding_glove
import torch
import pandas as pd
import cv2


# Test GloVe embedding function

glove_path_300 = 'Data/glove.6B/glove.6B.300d.txt'

class_label = 'wine_cellar'
subclass_label = 'barrel_storage' # '' for no cubclass
class_weight= 0.7

cl_label, cl_embedding= get_SUN_word_embedding_glove(glove_path_300, class_label, alpha=0.5)
sbcl_label, sbcl_embedding= get_SUN_word_embedding_glove(glove_path_300, subclass_label, alpha=0.5)
final_embedding= class_weight*cl_embedding+(1-class_weight)*sbcl_embedding
torch_embedding_300 = torch.from_numpy(final_embedding)
print(cl_label)
print(sbcl_label)
print(torch_embedding_300.shape)





# Test AwA2 Dataset structure
test_with = SUNDataset(dataset_path='Data/SUN/',
                        class_embedding_path='Data/SUN/class_all_embeddings.txt',
                        image_path='Data/SUN/images')

sample_with = test_with[100]
print('Sample: ', sample_with)
test_with.show_info()
#
test_without = SUNDataset(dataset_path='Data/SUN/',
                           class_embedding_path='Data/SUN/class_all_embeddings.txt',
                           image_path='Data/SUN/images',
                           use_predicates=False)

sample_without = test_without[100]
print('Sample: ', sample_without)
test_without.show_info()

# cv2.imshow('test', sample_with['image'])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

