import pandas as pd
import torch
from torch.utils.data import Dataset


class ZSLDataset(Dataset):
    """ Dataset class for ZSL Task """

    def __init__(self, dataset_path, transform=None, use_irevnet=False):
        """
        Args:
            dataset_path (string): Path to the folder of the dataset
            image_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.classes = pd.read_csv(dataset_path+'/classes.txt', index_col=0, header=0)
        self.labels = pd.read_csv(dataset_path+'/filenames_labels.txt', index_col=0, header=0)
        self.class_embeddings = pd.read_csv(dataset_path+'/glove_embeddings_300.txt', index_col=0, header=0)
        self.class_predicates = pd.read_csv(dataset_path+'/class_predicates.txt', header=0, index_col=0)

        if not use_irevnet:
            self.image_embeddings = pd.read_csv(dataset_path+'/resnet101_image_embeddings.txt', index_col=0, header=0)
        else:
            self.image_embeddings = pd.read_csv(dataset_path+'/irevnet_image_embeddings.txt', index_col=0, header=0)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename, class_id = self.labels.loc[idx]
        class_label = self.classes.loc[class_id].values[0]

        image_embedding = torch.tensor(self.image_embeddings.loc[idx].values)
        class_embedding = torch.tensor(self.class_embeddings.loc[class_label].values)
        class_predicate = torch.tensor(self.class_predicates.loc[class_id].values)

        sample = {'class_id': class_id,
                  'class_label': class_label,
                  'image_embedding': image_embedding,
                  'class_embedding': class_embedding,
                  'class_predicates': class_predicate
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def show_info(self):
        """ display information about dataset """
        print('\n')
        print('------------DATASET INFORMATION------------')
        print('N_Samples: ', self.__len__())
        print('N_Classes: ', len(self.classes))
        print('Image Embedding size: ', self.image_embeddings.shape[1])
        print('Class Embedding size: ', self.class_embeddings.shape[1], '({} classes found in word embedding)'.format(self.class_embeddings.shape[0]))
        print('N_predicates/attributes: ', self.class_predicates.shape[1])
        print('-------------------------------------------\n')
