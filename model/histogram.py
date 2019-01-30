
# coding: utf-8

# In[1]:


import re
import sys
import glob
import string
from pprint import pprint
from collections import Counter, OrderedDict

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
folder_path = "/home/niki/gcpEmailRecommendation/Scraping/debian_dataset/*"
file_name = BASE_PATH + "/model/dataframe3.csv"
file_name1 = BASE_PATH + "/model/dataframe4.csv"
file_name2 = BASE_PATH + "/model/dataframe5.csv"
sys.path.insert(0, BASE_PATH + '/Preprocessing')
PATH = BASE_PATH + '/model/first_model.pickle'

import preprocessing
import read_file
import datetime

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


# In[2]:


df_trn = pd.DataFrame()
df_tst = pd.DataFrame()
split_date = datetime.datetime.strptime('01 Sep 2017 23:01:14 +0000', '%d %b %Y %H:%M:%S %z')

users = []
dates  = []
trn_dates = []
tst_dates = []
trn_users = []
tst_users = []
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
        temp = deb_toppostremoval(temp)
        temp = deb_lemmatize(temp)
        temp = clean_debian(temp)
        if temp == '':
            cnt += 1
            continue
        users.append(sender)
        
        temp = obj.replace_tokens(temp)
        if flag==0:
            start_date = datetime.datetime.strptime(mail['Date'].split('(')[0].rstrip(),'%a, %d %b %Y %H:%M:%S %z')
            if start_date > split_date:
                df_tst = df_tst.append({'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':start_date}, ignore_index=True)
            else:
                df_trn = df_trn.append({'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':start_date}, ignore_index=True)
            t = temp
            flag = 1
            continue


        df = df.append({'body': str(t),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':datetime.datetime.strptime(mail['Date'].split('(')[0].rstrip(),'%a, %d %b %Y %H:%M:%S %z')}, ignore_index=True)
        
        if start_date <= split_date:
            t = t + temp
            df_trn = df_trn.append({'body': str(t),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':datetime.datetime.strptime(mail['Date'].split('(')[0].rstrip(),'%a, %d %b %Y %H:%M:%S %z')}, ignore_index=True)
        else:
            df_tst = df_tst.append({'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':datetime.datetime.strptime(mail['Date'].split('(')[0].rstrip(),'%a, %d %b %Y %H:%M:%S %z')}, ignore_index=True)       
    th_no += 1

#trn_users = list(df_trn.groupby("thread_no", as_index=False)['replier'].apply(lambda x: x.iloc[:-1]))
#tst_users = list(df_tst.groupby("thread_no", as_index=False)['replier'].apply(lambda x: x.iloc[:-1]))
#trn_dates = list(df_trn.groupby("thread_no", as_index=False)['cur_date'].apply(lambda x: x.iloc[:-1]))
#tst_dates = list(df_tst.groupby("thread_no", as_index=False)['cur_date'].apply(lambda x: x.iloc[:-1]))

print(cnt)
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
#pprint(rep_to_index)


for rep in df_trn['replier']:
    df_trn.loc[df_trn['replier']==rep,'int_replier'] = rep_to_index[rep]
#print(df_trn.head)    

for rep in df_tst['replier']:
    df_tst.loc[df_tst['replier']==rep,'int_replier'] = rep_to_index[rep]

for rep in df['replier']:
    df.loc[df['replier']==rep,'int_replier'] = rep_to_index[rep]
    
df_tst['replier'] = df_tst.groupby('thread_no')['replier'].shift(-1)
df_tst['int_replier'] = df_tst.groupby('thread_no')['int_replier'].shift(-1)

df_trn['replier'] = df_trn.groupby('thread_no')['replier'].shift(-1)
df_trn['int_replier'] = df_trn.groupby('thread_no')['int_replier'].shift(-1)

df_tst.dropna(inplace=True)
df_trn.dropna(inplace=True)

df_trn.to_csv(file_name)
df_tst.to_csv(file_name1)
df.to_csv(file_name2)


# In[3]:

tile = open('hist.txt','w')
qw = df.groupby(['replier']).size().reset_index(name='counts')
tile.write(str(qw.groupby(['counts']).size()))
tile.close()
plt.hist(qw['counts'],bins= 50,color='red')
plt.ylabel('Frequency')
plt.xlabel('User')
#plt.show()
plt.savefig('myfig.png')

