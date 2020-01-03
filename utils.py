import re
import numpy as np


def get_word_embedding_glove(file='', label='', alpha=0.5):
    """
        Args:
            file (string): glove embedding .txt file
            label (string): class in question
            alpha (float): mixture parameter (default=0.5), only used for composite label

        Returns:
            found_labels (list): labels of found words
            word_embedding (np.ndarray): GloVe representation of class, may be weighted average (alpha) if composite class
    """
    if not file:
        raise ValueError('invalid filepath')

    if not label:
        raise ValueError('no label provided')

    sub_labels = label.split('+')  # split in case of composite label

    found_labels = []
    word_embeddings = []
    for sub_label in sub_labels:
        found = False
        with open(file) as reader:
            for line in reader:
                if re.match(r'{0} '.format(sub_label), line):
                    line_split = line.rstrip().split(' ')
                    found_labels.append(line_split[0])
                    word_embeddings.append(np.asarray(line_split[1:], dtype=float))
                    found = True
        if not found:
            print('Could not find '+ sub_label)
            return found_labels, [] #return epty lists
    if not len(found_labels) == 1:
        if len(found_labels) == 2:
            word_embedding = (1 - alpha) * word_embeddings[0] + alpha * word_embeddings[1]
        if len(found_labels) == 3:
            word_embedding = (1 - alpha) * 0.5 * word_embeddings[0] + (1 - alpha) * 0.5 * word_embeddings[1] + alpha * word_embeddings[2]
        if len(found_labels) == 4:
            word_embedding = (1 - alpha) * 0.33 * word_embeddings[0] + (1 - alpha) * 0.33 * word_embeddings[1] + (1 - alpha) * 0.33 * word_embeddings[2] + alpha * word_embeddings[3]
    else:
        word_embedding = word_embeddings[0]

    return found_labels, word_embedding


def get_CUB_word_embedding_glove(file='', label='', alpha=0.5):
    """
        Args:
            file (string): glove embedding .txt file
            label (string): class in question
            alpha (float): mixture parameter (default=0.5), only used for composite label

        Returns:
            found_labels (list): labels of found words
            word_embedding (np.ndarray): GloVe representation of class, may be weighted average (alpha) if composite class
    """
    if not file:
        raise ValueError('invalid filepath')

    if not label:
        raise ValueError('no label provided')

    sub_labels = label.split('+')  # split in case of composite label
    synonyms = {'sayornis': 'phoebe', 'geococcyx': 'roadrunner', 'violetear': 'colibri', 'jaeger': 'skua'}

    found_labels = []
    word_embeddings = []
    for sub_label in sub_labels:
        if sub_label in synonyms.keys():
            sub_label = synonyms[sub_label]

        found = False
        with open(file) as reader:
            for line in reader:
                if re.match(r'{0} '.format(sub_label), line):
                    line_split = line.rstrip().split(' ')
                    found_labels.append(line_split[0])
                    word_embeddings.append(np.asarray(line_split[1:], dtype=float))
                    found = True
        if not found:
            print('Could not find '+ sub_label)
    if not len(found_labels) == 1:
        if len(found_labels) == 2:
            word_embedding = (1 - alpha) * word_embeddings[0] + alpha * word_embeddings[1]
        if len(found_labels) == 3:
            word_embedding = (1 - alpha) * 0.5 * word_embeddings[0] + (1 - alpha) * 0.5 * word_embeddings[1] + alpha * word_embeddings[2]
        if len(found_labels) == 4:
            word_embedding = (1 - alpha) * 0.33 * word_embeddings[0] + (1 - alpha) * 0.33 * word_embeddings[1] + (1 - alpha) * 0.33 * word_embeddings[2] + alpha * word_embeddings[3]
    else:
        word_embedding = word_embeddings[0]

    return found_labels, word_embedding

def get_SUN_word_embedding_glove(file='', label='', alpha=0.5):
    """
        Args:
            file (string): glove embedding .txt file
            label (string): class in question
            alpha (float): mixture parameter (default=0.5), only used for composite label
        Returns:
            found_labels (list): labels of found words
            word_embedding (np.ndarray): GloVe representation of class, may be weighted average (alpha) if composite class
    """
    if not file:
        raise ValueError('invalid filepath')

    if not label:
        raise ValueError('no label provided')

    synonyms = {'barndoor': 'barn_door', 'frontseat': 'front_seat', 'needleleaf': 'needle_leaf', 'oilrig': 'oil_rig', 'procenium':'proscenium', 'thriftshop':'thrift_shop','videostore':'video_store','kindergarden':'kindergarten'}
    if label in synonyms.keys():
            label = synonyms[label]
    sub_labels = label.split('_')  # split in case of composite label
    

    found_labels = []
    word_embeddings = []
    for sub_label in sub_labels:
        print(sub_label)
        if sub_label in synonyms.keys():
            sub_label = synonyms[sub_label]
        found = False
        with open(file, encoding="utf8") as reader:
            for line in reader:
                if re.match(r'{0} '.format(sub_label), line):
                    line_split = line.rstrip().split(' ')
                    found_labels.append(line_split[0])
                    word_embeddings.append(np.asarray(line_split[1:], dtype=float))
                    found = True
                    print('Yes')
        if not found:
            print('Could not find '+ sub_label)
            
    if not len(found_labels) == 1:
        if len(found_labels) == 2:
            word_embedding = (1 - alpha) * word_embeddings[0] + alpha * word_embeddings[1]
        if len(found_labels) == 3:
            word_embedding = (1 - alpha) * 0.5 * word_embeddings[0] + (1 - alpha) * 0.5 * word_embeddings[1] + alpha * word_embeddings[2]
        if len(found_labels) == 4:
            word_embedding = (1 - alpha) * 0.33 * word_embeddings[0] + (1 - alpha) * 0.33 * word_embeddings[1] + (1 - alpha) * 0.33 * word_embeddings[2] + alpha * word_embeddings[3]
    else:
        word_embedding = word_embeddings[0]

    return found_labels, word_embedding