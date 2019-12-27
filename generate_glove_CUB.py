
from utils import get_CUB_word_embedding_glove

import pandas as pd
glove_path_300 = 'Data/glove/glove.6B.300d.txt'

#generate glove embeddings for classes
dataset_path = 'Data/CUB_200_2011/CUB_200_2011/'
df = pd.read_csv(dataset_path+'classes.txt', delimiter=' ...\.', names=['class_id', 'label']).set_index('class_id')
df['label'] = df['label'].str.replace('_', '+')
df['label'] = df['label'].str.lower()

embeddings = []
class_names = []
synonyms = {'sayornis' : 'phoebe', 'geococcyx' : 'roadrunner', 'violetear' : 'colibri', 'jaeger' : 'skua'}
for i in range(0, len(df)):
    class_name = df.iloc[i]['label']

    found_labels_300, word_embedding_300 = get_CUB_word_embedding_glove(glove_path_300, class_name, alpha=0.5)
    embeddings.append(word_embedding_300)
    class_names.append(class_name)
    print(str(i) + ' / ' + str(len(df)))


out_df = pd.DataFrame(embeddings, columns=range(0,300))
out_df.insert(0, 'class_name', class_names)
out_df.to_csv('CUB_glove_embedding.txt', sep=',', index=False)