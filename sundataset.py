
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

class SUNDataset(Dataset):
    """Scene Understanding with attributes dataset"""

    def __init__(self, dataset_path, class_embedding_path, image_path, transform=None, use_predicates=True):
        """
        Args:
            dataset_path (string): Path to the folder of the dataset (classes.txt, class_predicates.txt, filenames_labels.txt)
            class_embedding_path (string): Path to the csv file with word embeddings of existing classes (label is id)
            image_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.classes = pd.read_csv(dataset_path+'class_label.txt', sep=" ")
        self.labels =pd.read_csv(dataset_path+'image_class_label.txt', sep=" ")
        self.class_embeddings = pd.read_csv(class_embedding_path, sep=" ",header =None)
        self.image_path = image_path
        self.transform = transform
        self.use_predicates = use_predicates
        if self.use_predicates:
            self.norm_attribute_list= pd.read_csv(dataset_path+'attribute_vector_class.txt', sep=",", header=None)
            self.attribute_list= pd.read_csv(dataset_path+'orig_attribute_vector_class.txt', sep=",", header=None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename, class_name, sub_class_name, class_id = self.labels.iloc[idx]

        #class_label = self.classes.loc[class_id].values[0]
        img_name = os.path.join(self.image_path, filename)
        image = cv2.imread(img_name)
        v=self.class_embeddings.iloc[class_id-1][2:].values
        #print(self.class_embeddings.iloc[class_id-1][2:].values[0].dtype)
        class_embedding = torch.tensor(v.astype(float))

        #class_label = self.classes.loc[class_id].values[0]

        if self.use_predicates:
            class_predicate = torch.tensor(self.norm_attribute_list.iloc[class_id-1].values)

            sample = {'class_id': class_id,'class_label': class_name,'sub_class_label': sub_class_name,'image': image,'attribute_vector': class_predicate,'class_embedding': class_embedding}
        else:
            sample = {'class_id': class_id,'class_label': class_name,'sub_class_label': sub_class_name,'image': image,'class_embedding': class_embedding}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def show_info(self):
        """ display information about dataset """
        print('\n')
        print('------------DATASET INFORMATION------------')
        print('N° Samples: ', self.__len__())
        print('N° Classes: ', len(self.classes))
        print('Class Embedding size: ', self.class_embeddings.shape[1]-2, '({} classes found in word embedding)'.format(self.class_embeddings.shape[0]))
        if self.use_predicates:
            print('N° predicates/attributes: ', len(self.attribute_list))
        print('-------------------------------------------\n')