import pandas as pd
import numpy as np
from utils import get_SUN_word_embedding_glove

data = pd.read_csv('Data/SUN/image_list.txt', sep=" ", header=None)
data.columns = ["image"]

cls_=[]
sub_cls=[]
for i in range(data.shape[0]):
    s=data.iloc[i][0]
    cnt=s.count('/')
    if(cnt ==3):
        s1=s.split('/')
        cls_.append(s1[1])
        sub_cls.append(s1[2])
    else:
        s1=s.split('/')
        cls_.append(s1[1])
        sub_cls.append('')

data['class']=cls_
data['sub_class']=sub_cls

df_class=pd.DataFrame()
df_class['class']=cls_
df_class['sub_class']=sub_cls

df_class=df_class.drop_duplicates()
idx=[]
for i in range(1,df_class.shape[0]+1):
    idx.append(i)
df_class['label']=idx

result = pd.merge(data, df_class,  how='left', left_on=['class','sub_class'], right_on = ['class','sub_class'])
# write result to image_class_label.txt
# write df_class to class_label.txt

wemb_path='Data/glove.6B/glove.6B.300d.txt'
cl_name=[]
cl_emb=[]
scl_name=[]
scl_emb=[]
tot_name=[]
tot_emb=[]
cl_wt=0.7
for i in range(0, df_class.shape[0]):
    #print(i)
    cl=df_class.iloc[i][0]
    scl=df_class.iloc[i][1]
    if scl:
        a,b1= get_SUN_word_embedding_glove(wemb_path,cl)
        cl_name.append(cl)
        cl_emb.append(b1)
        a,b2= get_SUN_word_embedding_glove(wemb_path,scl)
        scl_name.append(scl)
        scl_emb.append(b2)
        b3= cl_wt*b1+(1-cl_wt)*b2
        tot_name.append(scl)
        tot_emb.append(b3)
    else:
        a,b1= get_SUN_word_embedding_glove(wemb_path,cl)
        cl_name.append(cl)
        cl_emb.append(b1)
        tot_name.append('')
        tot_emb.append(b1)

df_cslemb=pd.DataFrame()
df_cslemb[0]=cl_name
df_cslemb[1]=tot_name
x=np.asarray(tot_emb)
x2=pd.DataFrame(x)
x3=pd.concat([df_cslemb,x2], axis=1)
#write to csv class_all_embeddings.txt
