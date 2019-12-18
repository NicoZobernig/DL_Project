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
        with open(file) as reader:
            for line in reader:
                if re.match(r'{0} '.format(sub_label), line):
                    line_split = line.rstrip().split(' ')
                    found_labels.append(line_split[0])
                    word_embeddings.append(np.asarray(line_split[1:], dtype=float))

    if not len(found_labels) == 1:
        word_embedding = (1 - alpha) * word_embeddings[0] + alpha * word_embeddings[1]
    else:
        word_embedding = word_embeddings[0]

    return found_labels, word_embedding
