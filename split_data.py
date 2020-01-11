import pandas as pd
import os


def split_data(dataset='', class_names_file='', new_dataset_folder_name=''):
    resnet_im_embeddings = pd.read_csv(dataset+'resnet101_image_embeddings.txt', index_col=0, header=0)
    irevnet_im_embeddings = pd.read_csv(dataset+'irevnet_image_embeddings.txt', index_col=0, header=0)

    files_labels = pd.read_csv(dataset+'filenames_labels.txt', index_col=0, header=0)
    classes = pd.read_csv(dataset+'classes.txt', index_col=0, header=0)
    classes_of_interest = pd.read_csv(class_names_file, header=None)
    row_oi = []
    for class_of_interest in classes_of_interest.iloc[:, 0].values:
        class_id = classes.loc[classes['label'] == class_of_interest].index.tolist()[0]
        class_row_ids = files_labels.loc[files_labels['class_id'] == class_id].index.tolist()
        row_oi.append(class_row_ids)

    row_ids = [item for sublist in row_oi for item in sublist]  # flatten list of
    row_ids.sort()

    if not os.path.exists(dataset+new_dataset_folder_name):
        os.mkdir(dataset+new_dataset_folder_name)

    new_im_emb = resnet_im_embeddings.iloc[row_ids, :]
    new_im_emb.index = range(new_im_emb.shape[0])
    new_im_emb.index.name = 'sample_id'
    new_im_emb.to_csv(dataset+new_dataset_folder_name+'resnet101_image_embeddings.txt')

    new_im_emb = irevnet_im_embeddings.iloc[row_ids, :]
    new_im_emb.index = range(new_im_emb.shape[0])
    new_im_emb.index.name = 'sample_id'
    new_im_emb.to_csv(dataset + new_dataset_folder_name + 'irevnet_image_embeddings.txt')

    new_file_labels = files_labels.iloc[row_ids, :]
    new_file_labels.index = range(new_file_labels.shape[0])
    new_file_labels.index.name = 'sample_id'
    new_file_labels.to_csv(dataset+new_dataset_folder_name+'filenames_labels.txt')


def main():
    split_data(dataset='ZSL_Data/AwA2/',
               class_names_file='ZSL_Data/AwA2/valclasses1.txt',
               new_dataset_folder_name='val_set_1/')
    print('done.')
    split_data(dataset='ZSL_Data/AwA2/',
               class_names_file='ZSL_Data/AwA2/valclasses2.txt',
               new_dataset_folder_name='val_set_2/')
    print('done.')
    split_data(dataset='ZSL_Data/AwA2/',
               class_names_file='ZSL_Data/AwA2/valclasses3.txt',
               new_dataset_folder_name='val_set_3/')
    print('done.')


if __name__ == '__main__':
    main()
