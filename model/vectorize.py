
# coding: utf-8

# In[1]:


from dask import dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count

nCores = cpu_count() - 1


# In[2]:


import re
import sys
import glob
import string
from pprint import pprint
from collections import Counter, OrderedDict

import networkx as nx
import operator


import spacy
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import pickle
import math

import gensim.models
import codecs

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)

BASE_PATH = '/home/niki/gcpEmailRecommendation'
doc2vec_path="/home/niki/apnews_dbow/doc2vec.bin"
folder_path = "/home/niki/Dataset_1_removed/*"
#folder_path = "/home/niki/gcpEmailRecommendation/Scraping/mini_deb/*"
file_name = "dataframe9_tfidf.csv"
file_name1 = "dataframe10_tfidf.csv"
file_name2 = "dataframe11_tfidf.csv"
sys.path.insert(0, BASE_PATH + '/Preprocessing')
#PATH = BASE_PATH + '/model/second_model.pickle'
TRAIN_PATH = 'train2.pkl'
TEST_PATH  = 'test2.pkl'
#TFIDF_PATH = 'tfidf.pkl'
USER_TRAIN = 'user_weights2.npy'
USER_TEST  = 'user_weights_test2.npy'
REM_PATH = 'users2.pkl'
USER_DICT = 'userdict.pkl'
MATRIX_DAT = BASE_PATH + '/matrix.dat'

import preprocessing
import read_file
import datetime

nile = open('debug.txt','w')

import random

def read_content_long_msg(content) :
# returns the minimized content for long msg body
    sentence_limit = 12
    content1 = ''
    lines = content.split('.')
    if len(lines) < sentence_limit :
        return content
    start = 0
    if lines[0].startswith('On') and lines[0].endswith('wrote:') :
        start = 1
    for x in range(start, start + 3) :
        content1 += lines[x]

    start = start + 4
    end = len(lines) - 4
    short_list = []
    for x in range(3) :
        short_list.append(random.randint(start, end))
    short_list.sort()
    for x in range(3) :
        content1 += lines[short_list[x]]
                                                                                                                        
    start = len(lines) - 4
    for x in range(start, start + 3) :
        content1 += lines[x]
    return content1

def read_content_noshort_msg(content) :
    # returns true if this msg body has to be taken, false otherwise
    sentence_limit = 3
    words_limit = 50
    lines = content.split('.')
    if len(lines) < sentence_limit :
        words = content.split(' ')
        if len(words) < words_limit :
            return False
    return True

def extract_debian(text):
    text = text.split('\n\n\n')
    header = text[0].split('\n')
    body = text[1]
    sender = header[2].split(':')[1].split('<')[0]
#     print('Sender',sender)
#     print('Body \n',body)
    return sender,body

def clean_debian(temp):
    temp = re.sub('\n+','\n',temp)
    temp = re.sub('\n',' ',temp)
    temp = re.sub('\t',' ',temp)
    temp = re.sub(' +',' ',temp)
    return temp

def deb_lemmatize(doc):        
    doc = nlp(doc)
    article, skl_texts = '',''
    for w in doc:
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
            article += " " + w.lemma_
        if w.text == '\n':                
            skl_texts += " " + article
            article = ''       
    return skl_texts

def deb_toppostremoval(temp):
    strings = temp.splitlines()
    temp = ''
    for st in strings:
        st = st.strip()
        if len(st)>0:
            if st[0]=='>':
                continue
            else:
                temp += '\n' + st
    return temp

df = pd.DataFrame()
folder = glob.glob(folder_path)
obj = preprocessing.preprocess()
count_file = 0
thread_list=[]
try:
    for fol in tqdm_notebook(folder):
        files = glob.glob(fol+'/*.txt')
        threads = []
        for file in files:
            ob = read_file.file_content(file)
            ob.read_file_content()
            threads.append(ob.mail)
            count_file += 1
        sorted_threads = sorted(threads, key=lambda ke: datetime.datetime.strptime(ke['Date'].split('(')[0].rstrip(),'%a, %d %b %Y %H:%M:%S %z'))
        thread_list.append(sorted_threads)
except:
    print(fol)
print(len(thread_list))
print(count_file)
nile.write(f'Threads : {len(thread_list)}\n')
nile.write(f'Mails : {count_file}\n')


# In[3]:

df_trn = pd.DataFrame()
df_tst = pd.DataFrame()
split_date = datetime.datetime.strptime('01 Sep 2017 23:01:14 +0000', '%d %b %Y %H:%M:%S %z')

dates  = []
trn_dates = []
tst_dates = []
trn_users = []
tst_users = []
qw1 = []
th_no = 0
cnt = 0

for thr in thread_list:
    start_date = ""
    flag = 0
    t = ''
    for mail in thr:
        temp = ''
        sender = mail['From'].split('<')[0].strip()
        temp   = mail['content']
        original = temp
        temp = deb_toppostremoval(temp)
        if read_content_noshort_msg(temp) == False :
            continue
        temp = read_content_long_msg(temp)
        temp = deb_lemmatize(temp)
        temp = clean_debian(temp)
        if temp == '':
            cnt += 1
            continue
        
        temp = obj.replace_tokens(temp)
        df = df.append({'mail': original,'body': str(t),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':datetime.datetime.strptime(mail['Date'].split('(')[0].rstrip(),'%a, %d %b %Y %H:%M:%S %z')}, ignore_index=True)
        if flag==0:
            start_date = datetime.datetime.strptime(mail['Date'].split('(')[0].rstrip(),'%a, %d %b %Y %H:%M:%S %z')
            if start_date > split_date:
                df_tst = df_tst.append({'mail': original,'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':start_date}, ignore_index=True)
            else:
                df_trn = df_trn.append({'mail': original,'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':start_date}, ignore_index=True)
            t = temp
            flag = 1
            continue
   
        qw1.append(sender)

        if start_date <= split_date:
            t = t + temp
            df_trn = df_trn.append({'mail': original,'body': str(t),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':datetime.datetime.strptime(mail['Date'].split('(')[0].rstrip(),'%a, %d %b %Y %H:%M:%S %z')}, ignore_index=True)
        else:
            df_tst = df_tst.append({'mail': original,'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':datetime.datetime.strptime(mail['Date'].split('(')[0].rstrip(),'%a, %d %b %Y %H:%M:%S %z')}, ignore_index=True)       
    th_no += 1

qw = df.groupby(['replier']).size().reset_index(name='counts')
qw = qw.sort_values(by='counts',ascending=0)
users = list(qw.drop(qw[(qw.counts < 2) & (df['replier'].isin(qw1))].index)['replier'])
print(qw.head())
qw = qw.drop(qw[(qw.counts > 1) & (df['replier'].isin(qw1))].index)
qw.to_pickle(REM_PATH)

rem_users = list(qw['replier'])
print('Removed users :',len(rem_users),'\n')
nile.write(f'Removed users :{len(rem_users)}\n')

print('BEFORE')
print('Train : ',df_trn.shape[0])
print('Test : ',df_tst.shape[0],'\n')
nile.write(f'BEFORE\n')
nile.write(f'Train : {df_trn.shape[0]}\n')
nile.write(f'Test : {df_tst.shape[0]}\n')
df_trn = df_trn[~df_trn['replier'].isin(rem_users)]
df_tst = df_tst[~df_tst['replier'].isin(rem_users)]
print('AFTER')
print('Train : ',df_trn.shape[0])
print('Test : ',df_tst.shape[0],'\n')


nile.write(f'AFTER\n')
nile.write(f'Train : {df_trn.shape[0]}\n')
nile.write(f'Test : {df_tst.shape[0]}\n')


print(len(users))


trn_users = list(df_trn.groupby("thread_no", as_index=False)['replier'].apply(lambda x: x.iloc[:-1]))
tst_users = list(df_tst.groupby("thread_no", as_index=False)['replier'].apply(lambda x: x.iloc[:-1]))
trn_dates = list(df_trn.groupby("thread_no", as_index=False)['cur_date'].apply(lambda x: x.iloc[:-1]))
tst_dates = list(df_tst.groupby("thread_no", as_index=False)['cur_date'].apply(lambda x: x.iloc[:-1]))

print(cnt)
nile.write(f'Null files : {cnt}\n')
print(count_file)
print(len(df['body']))
print(len(df['thread_no'].unique()))
print(len(df['replier'].unique()))

rep_to_index = {}
index = 0
for rep in users:
    if rep_to_index.get(rep, 0) == 0:
        rep_to_index[rep] = index
        index += 1

jimm = open(USER_DICT,'wb')
pickle.dump(rep_to_index,jimm)
jimm.close()

# pprint(rep_to_index)
nile.write('\n\n\n\n')
nile.write(f'{rep_to_index}\n\n')

for rep in df_trn['replier']:
    df_trn.loc[df_trn['replier']==rep,'int_replier'] = rep_to_index[rep]
#print(df_trn.head)    

for rep in df_tst['replier']:
    df_tst.loc[df_tst['replier']==rep,'int_replier'] = rep_to_index[rep]

#for rep in df['replier']:
#    df.loc[df['replier']==rep,'int_replier'] = rep_to_index[rep]
    
df_tst['replier'] = df_tst.groupby('thread_no')['replier'].shift(-1)
df_tst['int_replier'] = df_tst.groupby('thread_no')['int_replier'].shift(-1)

df_trn['replier'] = df_trn.groupby('thread_no')['replier'].shift(-1)
df_trn['int_replier'] = df_trn.groupby('thread_no')['int_replier'].shift(-1)


df_tst.dropna(inplace=True)
df_trn.dropna(inplace=True)

df_trn.to_csv(file_name)
df_tst.to_csv(file_name1)
df.to_csv(file_name2)



# In[4]:


# words = Counter()
# for sent in df_trn.body.values:
#     words.update(w.text.lower() for w in nlp(sent))

# words = sorted(words, key=words.get, reverse=True)
# #print(words)
# words = ['_PAD','_UNK'] + words

# word2idx = {o:i for i,o in enumerate(words)}
# idx2word = {i:o for i,o in enumerate(words)}

# #@jit
# def indexer(s):
#     start_alpha=0.01
#     infer_epoch=1000
#     m = gensim.models.Doc2Vec.load(doc2vec_path)
#     Document_vector = [x for x in m.infer_vector(s, alpha=start_alpha, steps=infer_epoch)]
#     print('we are working')
#     return Document_vector


# In[5]:


np.set_printoptions(threshold = sys.maxsize)
user_indices = []
trn_user_indices = []
tst_user_indices = []

for u in users:
    user_indices.append(rep_to_index[u])

for v in trn_users:
    trn_user_indices.append(rep_to_index[v])

for w in tst_users:
    tst_user_indices.append(rep_to_index[w])


# In[6]:


user_vec_len = len(rep_to_index) + 1
print("User vec len: "+str(user_vec_len))

# In[7]:

interaction_matrix = np.load(MATRIX_DAT)
print("mtrxshape")
print(interaction_matrix.shape)

G=nx.from_numpy_matrix(interaction_matrix, parallel_edges=False, create_using=nx.DiGraph())
G.edges(data=True)

index_list=[]
for i in range(0, user_vec_len):
    index_list.append(i)

import math
indexx=0
weight_list = []
#print(len(trn_dates))
nile.write(f'Train users : {len(trn_users)}\n')
#print(df_trn.thread_no.shape[0])
#print(len(trn_user_indices))

thread_no_list = list(df_trn['thread_no'])

for i in range(0, len(df_trn.groupby("thread_no"))):
    temp_index=indexx
    thread_start_date = trn_dates[temp_index].to_pydatetime()
    array  = np.zeros(user_vec_len)
    if temp_index < df_trn.thread_no.shape[0]:
        print('***',temp_index, temp_index + list(df_trn.thread_no).count(thread_no_list[temp_index]))
        for j in range(temp_index, temp_index + list(df_trn.thread_no).count(thread_no_list[temp_index])):
            if j>temp_index:
                cur_date = trn_dates[j].to_pydatetime()
                date_diff = cur_date - thread_start_date
                total_seconds = date_diff.total_seconds()
                #print(str(total_seconds))
                # decay_value = math.exp(-total_seconds)
                # print(str(decay_value))
            array[trn_user_indices[j]] = 1
            
            personal_df = pd.DataFrame({'user_index':index_list, 'prev_participants':list(array)})

            personalization = personal_df.set_index('user_index')['prev_participants'].to_dict()
            ppr = nx.pagerank(G, alpha = 0.8, personalization = personalization)
            
            ppr_value = ppr[trn_user_indices[j]]
            print(ppr_value)
            array[trn_user_indices[j]] = ppr_value

            weight_list.append(list(array))
            indexx+=1

trn_weights = np.array(weight_list)


# In[8]:


import math
indexx=0
weight_list = []
#print(len(tst_dates))
nile.write(f'Test users : {len(tst_users)}\n')
#print(df_tst.thread_no.shape[0])
#print(len(tst_user_indices))

thread_no_list = list(df_tst['thread_no'])

for i in range(0, len(df_tst.groupby("thread_no"))):
    temp_index=indexx
    thread_start_date = tst_dates[temp_index].to_pydatetime()
    array  = np.zeros(user_vec_len)
    if temp_index < df_tst.thread_no.shape[0]:
        for j in range(temp_index, temp_index + list(df_tst.thread_no).count(thread_no_list[temp_index])):
            if j>temp_index:
                cur_date = tst_dates[j].to_pydatetime()
                date_diff = cur_date - thread_start_date
                total_seconds = date_diff.total_seconds()
                #print(str(total_seconds))
                # decay_value = math.exp(-total_seconds)
                # print(str(decay_value))
            array[tst_user_indices[j]] = 1

            personal_df = pd.DataFrame({'user_index':index_list, 'prev_participants':list(array)})

            personalization = personal_df.set_index('user_index')['prev_participants'].to_dict()
            ppr = nx.pagerank(G, alpha = 0.8, personalization = personalization)
            
            ppr_value = ppr[trn_user_indices[j]]
            print(ppr_value)
            array[trn_user_indices[j]] = ppr_value

            weight_list.append(list(array))
            indexx+=1

tst_weights = np.array(weight_list)
#print(tst_weights)


# In[14]:


print(trn_weights.shape)
print(type(trn_weights))

# trn_weights.tofile('/home/niki/user_weights.dat')
#np.save(USER_TRAIN, trn_weights)  
#np.save(USER_TEST, tst_weights)  


np.save(USER_TRAIN, trn_weights)  
np.save(USER_TEST, tst_weights)  



# In[10]:


# embedding |> flag
class VectorizeData(Dataset):
    def __init__(self, df_path, maxlen=1000):
        self.df = pd.read_csv(df_path, error_bad_lines=False)
        self.df['body'] = self.df.body.apply(lambda x: str(x).strip())
        print('Indexing...')

        tfidf = TfidfVectorizer(lowercase = True, max_features = 1000, max_df = 0.85)
        self.df['bodyidx'] = list(tfidf.fit_transform(self.df['body']).toarray())
        
        print('Calculating lengths')
        self.df['lengths'] = self.df.bodyidx.apply(len)
        self.maxlen = 1000
        print(self.df)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        X = self.df.bodyidx[idx]
        lens = self.df.lengths[idx]
        y = self.df.int_replier[idx]
        return X,y,lens
    
    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded


# In[11]:



#ds = VectorizeData(file_name2)
#dtrain = VectorizeData(file_name)
#dtest = VectorizeData(file_name1)

dtrain = VectorizeData(file_name)
dtest = VectorizeData(file_name1)



# In[12]:


dtrain.df.to_pickle(TRAIN_PATH)
dtest.df.to_pickle(TEST_PATH)
#ds.df.to_pickle(TFIDF_PATH)

# In[13]:
nile.close()

print(user_vec_len)
