
# coding: utf-8

# # Preprocessing, building a Pandas dataframe and saving it as a  .csv file

# In[5]:
from dask import dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count
nCores = cpu_count() - 1

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

torch.manual_seed(42)	

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
        sorted_threads = sorted(threads, key=lambda ke: datetime.datetime.strptime(ke['Date'],'%a, %d %b %Y %H:%M:%S %z'))
        thread_list.append(sorted_threads)
except:
    print(fol)
print(len(thread_list))
print(len(count_file))


# In[6]:


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
            start_date = datetime.datetime.strptime(mail['Date'],'%a, %d %b %Y %H:%M:%S %z')
            if start_date > split_date:
                df_tst = df_tst.append({'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':start_date}, ignore_index=True)
            else:
                df_trn = df_trn.append({'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':start_date}, ignore_index=True)
            t = temp
            flag = 1
            continue


        df = df.append({'body': str(t),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':datetime.datetime.strptime(mail['Date'],'%a, %d %b %Y %H:%M:%S %z')}, ignore_index=True)
        
        if start_date <= split_date:
            t = t + temp
            df_trn = df_trn.append({'body': str(t),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':datetime.datetime.strptime(mail['Date'],'%a, %d %b %Y %H:%M:%S %z')}, ignore_index=True)
        else:
            df_tst = df_tst.append({'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date, 'cur_date':datetime.datetime.strptime(mail['Date'],'%a, %d %b %Y %H:%M:%S %z')}, ignore_index=True)       
    th_no += 1

trn_users = list(df_trn.groupby("thread_no", as_index=False)['replier'].apply(lambda x: x.iloc[:-1]))
tst_users = list(df_tst.groupby("thread_no", as_index=False)['replier'].apply(lambda x: x.iloc[:-1]))
trn_dates = list(df_trn.groupby("thread_no", as_index=False)['cur_date'].apply(lambda x: x.iloc[:-1]))
tst_dates = list(df_tst.groupby("thread_no", as_index=False)['cur_date'].apply(lambda x: x.iloc[:-1]))

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
pprint(rep_to_index)


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


# ## Indexing of words in vocab

# In[7]:


words = Counter()
for sent in df_trn.body.values:
    words.update(w.text.lower() for w in nlp(sent))

words = sorted(words, key=words.get, reverse=True)
print(words)
words = ['_PAD','_UNK'] + words

word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

def indexer(s):
    start_alpha=0.01
    infer_epoch=1000
    m = gensim.models.Doc2Vec.load(doc2vec_path)
    Document_vector = [x for x in m.infer_vector(s, alpha=start_alpha, steps=infer_epoch)]
    return Document_vector


# # User Vector - construction

# In[8]:


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


# In[9]:


user_vec_len = max(user_indices) + 1


# In[10]:


import math
indexx=0
weight_list = []
print(len(trn_dates))
print(len(trn_users))
print(df_trn.thread_no.shape[0])
print(len(trn_user_indices))

thread_no_list = list(df_trn['thread_no'])

for i in range(0, len(df_trn.groupby("thread_no"))):
    temp_index=indexx
    thread_start_date = trn_dates[temp_index].to_pydatetime()
    array  = np.zeros(user_vec_len)
    if temp_index < df_trn.thread_no.shape[0]:
        for j in range(temp_index, temp_index + list(df_trn.thread_no).count(thread_no_list[temp_index])):
            if j>temp_index:
                cur_date = trn_dates[j].to_pydatetime()
                date_diff = cur_date - thread_start_date
                total_seconds = date_diff.total_seconds()
                print(str(total_seconds))
                decay_value = math.exp(-total_seconds)
                print(str(decay_value))
            array[trn_user_indices[j]] = 1
            weight_list.append(list(array))
            indexx+=1

trn_weights = np.array(weight_list)
print(trn_weights)


# In[11]:


import math
indexx=0
weight_list = []
print(len(tst_dates))
print(len(tst_users))
print(df_tst.thread_no.shape[0])
print(len(tst_user_indices))

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
                print(str(total_seconds))
                decay_value = math.exp(-total_seconds)
                print(str(decay_value))
            array[tst_user_indices[j]] = 1
            weight_list.append(list(array))
            indexx+=1

tst_weights = np.array(weight_list)
print(tst_weights)


# # Dataset Loading

# In[12]:


# embedding |> flag
class VectorizeData(Dataset):
    def __init__(self, df_path, maxlen=300):
        self.df = pd.read_csv(df_path, error_bad_lines=False)
        self.df['body'] = self.df.body.apply(lambda x: x.strip())
        print('Indexing...')
        self.df['bodyidx'] = dd.from_pandas(self.df,npartitions=nCores).\
        map_partitions(
          lambda dp : dp.apply(
             lambda x : indexer(x.body),axis=1)).\
        compute(get=get)
        print('Calculating lengths')
        self.df['lengths'] = self.df.bodyidx.apply(len)
#         if calc_maxlen == True:
#             self.maxlen = max(self.df['lengths'])
#         else:
#             self.maxlen = maxlen
        self.maxlen = 300
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


# In[13]:


#ds = VectorizeData(file_name2)
dtrain = VectorizeData(file_name)
dtest = VectorizeData(file_name1)

dtrain.df.to_csv('/home/niki/train.csv')


# # Pytorch Feedforward Neural Network model

# In[15]:


input_size = dtrain.maxlen
hidden_size = 50
num_classes = user_vec_len
num_epochs = 5
batch_size = 1
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[16]:


print(user_vec_len)


# In[17]:


# embeddings |>
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size,user_vec_len, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size + user_vec_len, hidden_size)
        nn.init.xavier_uniform_(self.fc1.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias,0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        nn.init.xavier_uniform_(self.fc2.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias,0)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.logprob = nn.LogSoftmax(dim=1)  
    
    def forward(self, x,w):
#         x = torch.FloatTensor(x)
        catt = torch.cat((x,w),1)
        out = self.fc1(catt)
        out = self.relu(out)       
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.logprob(out)
        return out

# In[18]:


model = NeuralNet(input_size, hidden_size,user_vec_len, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate) 


# In[19]:


train_dl= DataLoader(dtrain, batch_size=1)
num_batch = len(train_dl)
for epoch in range(num_epochs):
    y_true_train = list()
    y_pred_train = list()
    total_loss_train = 0
    t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
    for we, w in zip(t,trn_weights):
        X = we[0]
        y = we[1]
        lengths = we[2]
        
        w = w.reshape(-1,1)
        w = w.transpose()
        
        X = np.array(X)
        X = X.reshape(-1,1)
        X = X.transpose()
        
        w = Variable(torch.Tensor(w).cpu())
        X = Variable(torch.Tensor(X).cpu())
        y = Variable(y.cpu())
        lengths = lengths.numpy()

        
        X = X.float()
        w = w.float()
        y = y.long()
        pred = model(X,w)
        # F.nll_loss can be replaced with criterion
        loss = F.nll_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        t.set_postfix(loss=loss.item())
        pred_idx = torch.max(pred, dim=1)[1]
        # print(y)
        print(pred_idx)
        # print('%%%%')

        y_true_train += list(y.cpu().data.numpy())
        y_pred_train += list(pred_idx.cpu().data.numpy())
        total_loss_train += loss.item()
        

    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_loss = total_loss_train/len(train_dl)
    print(f' Epoch {epoch}: Train loss: {train_loss} acc: {train_acc}')
torch.save(model.state_dict(),PATH)

# architecture 
# loading pickle file and predict


# # Testing 

# In[22]:


test_dl= DataLoader(dtest, batch_size=1)
num_batches = len(test_dl)
y_true_test1 = list()
y_pred_test1 = list()
total_loss_test = 0
tt = tqdm_notebook(iter(test_dl), leave=False, total=num_batches)
for we, w in zip(tt,tst_weights):
    X = we[0]
    y = we[1]
    lengths = we[2]
    
    w = w.reshape(-1,1)
    w = w.transpose()
    
    X = np.array(X)
    X = X.reshape(-1,1)
    X = X.transpose()

    w = Variable(torch.Tensor(w).cpu())
    X = Variable(torch.Tensor(X).cpu())
    y = Variable(y.cpu())
    lengths = lengths.numpy()

    X = X.float()
    w = w.float()
    y = y.long()
    pred = model(X,w)
    loss = F.nll_loss(pred, y)

    pred_idx = torch.max(pred, dim=1)[1]
    # print(y)
    print(pred_idx)
    # print('####')

    y_true_test1 += list(y.cpu().data.numpy())
    y_pred_test1 += list(pred_idx.cpu().data.numpy())
    print(loss.item())
    total_loss_test += loss.item()

train_acc = accuracy_score(y_true_test1, y_pred_test1)
train_loss = total_loss_test/len(test_dl)
print(f'Train loss: {train_loss} acc: {train_acc}')

