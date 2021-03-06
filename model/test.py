
# coding: utf-8

# In[1]:


import re
import sys
import glob
import string
from pprint import pprint
from collections import Counter, OrderedDict
from collections import defaultdict

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

BASE_PATH = '/home/niki'
PATH = BASE_PATH + '/0model.pt1model.pt2model.pt3model.pt4model.pt5model.pt6model.pt7model.pt8model.pt9model.pt10model.pt11model.pt12model.pt13model.pt14model.pt15model.pt16model.pt17model.pt18model.pt19model.pt'

TEST_PATH  = '/home/niki/test2.pkl'
USER_TEST  = '/home/niki/user_weights_test2.npy'
REM_PATH = '/home/niki/users2.pkl'
THREAD_DICT = 'thread_dict.pkl'
nile = open('debug-test.txt','w')
USER_LIMIT = 30

# In[2]:


user_vec_len = 1773


# In[5]:


tst_weights = np.load(USER_TEST)


# In[6]:


# embedding |> flag
class VectorizeData(Dataset):
    def __init__(self, df_path, maxlen=300):
        self.df = pd.read_pickle(df_path)
        self.maxlen = 1000
        print(self.df)
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = self.df.bodyidx[idx]
        lens = self.df.lengths[idx]
        y = self.df.int_replier[idx]
        tid = self.df.thread_no[idx]
        return X,y,lens,tid

    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded


# In[7]:


dtest = VectorizeData(TEST_PATH)


# In[8]:


input_size = 1000
hidden_size = 50
num_classes = user_vec_len
num_epochs = 5
batch_size = 1
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[9]:


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


# In[10]:


model = NeuralNet(input_size, hidden_size,user_vec_len, num_classes).to(device)
# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# opt = torch.optim.Adam(model.parameters(), lr=learning_rate) 
model.load_state_dict(torch.load(PATH))
model.eval()


# In[11]:

thread_map = defaultdict(list)
test_dl= DataLoader(dtest, batch_size=1)
num_batches = len(test_dl)
y_true_test1 = list()
y_pred_test1 = list()
total_loss_test = 0
hit = 0
tt = tqdm_notebook(iter(test_dl), leave=False, total=num_batches)
for we, w in zip(tt,tst_weights):
    X = we[0]
    y = we[1]
    lengths = we[2]
    thread_id = we[3]

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

    pred = pred.sort()
    array = pred[1][0][-20:]
    # thread_map[thread_id].extend(array)
    print(thread_id)
    print("ARRAY")
    print(array)
    for i in array:
        if i not in thread_map[thread_id]:
            if len(thread_map[thread_id]) + 1> limit :
                thread_map[thread_id].pop(0)
                thread_map[thread_id].append(i)
            else :
                thread_map[thread_id].append(i)
            # print(thread_map[thread_id])
    if y in array:
    	hit += 1


    y_true_test1 += list(y.cpu().data.numpy())
    y_pred_test1 += list(pred_idx.cpu().data.numpy())
    print(loss.item())
    total_loss_test += loss.item()
    break

accuracy = float(hit)/float(len(test_dl))
train_acc = accuracy_score(y_true_test1, y_pred_test1)
train_loss = total_loss_test/len(test_dl)
print(f'Test loss: {train_loss}')
print('Accuracy : ',accuracy)
print('\n')
nile.write(f'Test loss: {train_loss}\n')
nile.write(f'Accuracy : {accuracy}')
nile.write(f'\n\n')

# print(thread_map)
riley = open(THREAD_DICT,'wb')
pickle.dump(thread_map,riley)
riley.close()

