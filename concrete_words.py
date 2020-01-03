import pandas as pd
from utils import get_word_embedding_glove

glove_path_300 = 'Data/glove/glove.6B.300d.txt'

N = 1000


#http://crr.ugent.be/papers/Brysbaert_Warriner_Kuperman_BRM_Concreteness_ratings.pdf
df = pd.read_csv('Data/conc_words.csv',  header=0)

monograms = df['Bigram'] == 0
df = df[monograms]

df = df.sort_values('conc_score', ascending=0)



words = df['Word'].tolist()
embedding_list = []
found_words = []
i=1
for word in words:

    print('{} / {}'.format(i, N))
    found_label, word_embedding = get_word_embedding_glove(glove_path_300, word)
    if len(found_label) > 0:
        i += 1
        embedding_list.append(word_embedding)
        found_words.append(word)
    if len(found_words) >= N:
        break


out_df = pd.DataFrame({'word': found_words, 'embedding': embedding_list})

out_df.to_csv('Data/concrete_word_embeddings.txt', index=0)