import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class ZSLDataset(Dataset):
    """ Dataset class for ZSL Task """

    def __init__(self, dataset_path, image_path, transform=None, use_predicates=True):
        """
        Args:
            dataset_path (string): Path to the folder of the dataset
            image_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.classes = pd.read_csv(dataset_path+'/classes.txt', index_col=0, header=0)
        self.labels = pd.read_csv(dataset_path+'/filenames_labels.txt', index_col=0, header=0)
        self.class_embeddings = pd.read_csv(dataset_path+'/glove_embeddings_300.txt', index_col=0, header=0)
        self.image_embeddings = pd.read_csv(dataset_path+'/resnet101_image_embeddings.txt', index_col=0, header=0)
        self.image_path = image_path
        self.transform = transform
        self.use_predicates = use_predicates
        if self.use_predicates:
            self.class_predicates = pd.read_csv(dataset_path+'/class_predicates.txt', header=0, index_col=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename, class_id = self.labels.loc[idx]
        class_label = self.classes.loc[class_id].values[0]

        # img_name = os.path.join(self.image_path, filename)
        # image = cv2.imread(img_name)
        image_embedding = torch.tensor(self.image_embeddings.loc[idx].values)
        class_embedding = torch.tensor(self.class_embeddings.loc[class_label].values)

        if self.use_predicates:
            class_predicate = torch.tensor(self.class_predicates.loc[class_id].values)
            sample = {'class_id': class_id,
                      'class_label': class_label,
                      'image_embedding': image_embedding,
                      'class_embedding': class_embedding,
                      'class_predicates': class_predicate
                      }
        else:
            sample = {'class_id': class_id,
                      'class_label': class_label,
                      'image_embedding': image_embedding,
                      'class_embedding': class_embedding
                      }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def show_info(self):
        """ display information about dataset """
        print('\n')
        print('------------DATASET INFORMATION------------')
        print('N° Samples: ', self.__len__())
        print('N° Classes: ', len(self.classes))
        print('Image Embedding size: ', self.image_embeddings.shape[1])
        print('Class Embedding size: ', self.class_embeddings.shape[1], '({} classes found in word embedding)'.format(self.class_embeddings.shape[0]))
        if self.use_predicates:
            print('N° predicates/attributes: ', self.class_predicates.shape[1])
        print('-------------------------------------------\n')

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
#
# class AwA2Dataset(Dataset):
#     """Animals with Attributes dataset."""
#
#     def __init__(self, dataset_path, class_embedding_path, image_path, transform=None, use_predicates=True):
#         """
#         Args:
#             dataset_path (string): Path to the folder of the dataset (classes.txt, class_predicates.txt, filenames_labels.txt)
#             class_embedding_path (string): Path to the csv file with word embeddings of existing classes (label is id)
#             image_path (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#
#         self.classes = pd.read_csv(dataset_path+'classes.txt', index_col=0, header=0)
#         self.labels = pd.read_csv(dataset_path+'filenames_labels.txt', index_col=0, header=0)
#         self.class_embeddings = pd.read_csv(class_embedding_path, index_col=0, header=0)
#         self.image_path = image_path
#         self.transform = transform
#         self.use_predicates = use_predicates
#         if self.use_predicates:
#             self.class_predicates = pd.read_csv(dataset_path+'class_predicates.txt', header=0, index_col=0)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         filename, class_id = self.labels.iloc[idx]
#         class_label = self.classes.loc[class_id].values[0]
#
#         img_name = os.path.join(self.image_path, filename)
#         image = cv2.imread(img_name)
#         class_embedding = torch.tensor(self.class_embeddings.loc[class_label].values)
#
#         if self.use_predicates:
#             class_predicate = torch.tensor(self.class_predicates.loc[class_id].values)
#             sample = {'class_id': class_id,
#                       'class_label': class_label,
#                       'image': image,
#                       'class_predicates': class_predicate,
#                       'class_embedding': class_embedding
#                       }
#         else:
#             sample = {'class_id': class_id,
#                       'class_label': class_label,
#                       'image': image,
#                       'class_embedding': class_embedding
#                       }
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
#     def show_info(self):
#         """ display information about dataset """
#         print('\n')
#         print('------------DATASET INFORMATION------------')
#         print('N° Samples: ', self.__len__())
#         print('N° Classes: ', len(self.classes))
#         print('Class Embedding size: ', self.class_embeddings.shape[1], '({} classes found in word embedding)'.format(self.class_embeddings.shape[0]))
#         if self.use_predicates:
#             print('N° predicates/attributes: ', self.class_predicates.shape[1])
#         print('-------------------------------------------\n')
#
#
# class CUBDataset(Dataset):
#     """Caltech-UCSD Birds-200-2011 dataset."""
#
#     def __init__(self, dataset_path, class_embedding_path, image_path, transform=None, use_predicates=True):
#         """
#         Args:
#             dataset_path (string): Path to the folder of the dataset (classes.txt, class_predicates.txt, filenames_labels.txt)
#             class_embedding_path (string): Path to the csv file with word embeddings of existing classes (label is id)
#             image_path (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#
#         df = pd.read_csv(dataset_path+'classes.txt', delimiter=' ...\.', names=['class_id', 'label'], engine='python').set_index('class_id')
#         df['label'] = df['label'].str.replace('_', '+')
#         df['label'] = df['label'].str.lower()
#         self.classes = df
#         self.labels = pd.read_csv(dataset_path+'image_class_labels.txt', delimiter=' ',names=['image_id', 'label']).set_index('image_id')
#         self.images = pd.read_csv(dataset_path+'images.txt', delimiter=' ', names=['image_id', 'filename']).set_index('image_id')
#         self.class_embeddings = pd.read_csv(class_embedding_path, index_col=0, header=0)
#         self.image_path = image_path
#         self.transform = transform
#         self.use_predicates = use_predicates
#         if self.use_predicates:
#             self.class_predicates = pd.read_csv(dataset_path+'attributes/class_attribute_labels_continuous.txt', delimiter=' ',  names=range(1,313))
#             self.class_predicates.insert(0, 'class_id', range(1,len(self.classes)+1))
#             self.class_predicates = self.class_predicates.set_index('class_id')
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#
#         class_id = self.labels.iloc[idx]
#         class_label = self.classes.loc[class_id].values[0]
#         filename = self.images.iloc[idx].values[0]
#
#         img_name = os.path.join(self.image_path, filename)
#         image = cv2.imread(img_name)
#         class_embedding = torch.tensor(self.class_embeddings.loc[class_label].values[0])
#
#         if self.use_predicates:
#             class_predicate = torch.tensor(self.class_predicates.loc[class_id].values[0])
#             sample = {'class_id': class_id,
#                       'class_label': class_label,
#                       'image': image,
#                       'class_predicates': class_predicate,
#                       'class_embedding': class_embedding
#                       }
#         else:
#             sample = {'class_id': class_id,
#                       'class_label': class_label,
#                       'image': image,
#                       'class_embedding': class_embedding
#                       }
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
#     def show_info(self):
#         """ display information about dataset """
#         print('\n')
#         print('------------DATASET INFORMATION------------')
#         print('N° Samples: ', self.__len__())
#         print('N° Classes: ', len(self.classes))
#         print('Class Embedding size: ', self.class_embeddings.shape[1], '({} classes found in word embedding)'.format(self.class_embeddings.shape[0]))
#         if self.use_predicates:
#             print('N° predicates/attributes: ', self.class_predicates.shape[1])
#         print('-------------------------------------------\n')
