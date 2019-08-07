
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


# In[16]:
TRAIN_PATH = '/home/niki/train1.pkl'
USER_TRAIN = '/home/niki/user_weights1.npy'
BASE_PATH = '/home/niki'
PATH = BASE_PATH + '/'
REM_PATH = '/home/niki/users.pkl'

user_vec_len = 659

nile = open('debug-train.txt','w')
# In[2]:


trn_weights = np.load(USER_TRAIN)


# In[3]:


# embedding |> flag
class VectorizeData(Dataset):
	def __init__(self, df_path, maxlen=300):
		self.df = pd.read_pickle(df_path)
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


# In[4]:


dtrain = VectorizeData(TRAIN_PATH)


# In[9]:


input_size = 300
hidden_size = 50
num_layers = 1
dropout_p = 0.1
num_classes = user_vec_len
num_epochs = 20 
batch_size = 1
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[13]:


# embeddings |>
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size,user_vec_len, num_layers,dropout_p):
		super(NeuralNet, self).__init__()
		# self.fc1 = nn.Linear(input_size + user_vec_len, hidden_size)
		# nn.init.xavier_uniform_(self.fc1.weight,gain=nn.init.calculate_gain('relu'))
		# nn.init.constant_(self.fc1.bias,0)
		# self.relu = nn.ReLU()
		# self.fc2 = nn.Linear(hidden_size,hidden_size)
		# nn.init.xavier_uniform_(self.fc2.weight,gain=nn.init.calculate_gain('relu'))
		# nn.init.constant_(self.fc2.bias,0)
		# self.relu = nn.ReLU()
		# self.fc3 = nn.Linear(hidden_size, num_classes)
		# self.logprob = nn.LogSoftmax(dim=1)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
         
    
	def forward(self, x,w):
#         x = torch.FloatTensor(x)
		catt = torch.cat((x,w),1)
		out = self.lstm(catt)
		out = self.dropout(out)       
		out = self.fc1(out)
		#out = self.relu(out)
		out = self.fc2(out)
		#out = self.logprob(out)
		return out


# In[17]:


model = NeuralNet(input_size, hidden_size,user_vec_len, num_layers,dropout_p).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate) 


# In[18]:

model_name = ''
train_dl= DataLoader(dtrain, batch_size=1)
num_batch = len(train_dl)
for epoch in range(num_epochs):

    hit = 0
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
        pred = pred.sort()
        array = pred[1][0][-3:]
        if y in array:
        	hit += 1

        y_true_train += list(y.cpu().data.numpy())
        y_pred_train += list(pred_idx.cpu().data.numpy())
        total_loss_train += loss.item()

      
    accuracy = float(hit)/float(len(train_dl))
    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_loss = total_loss_train/len(train_dl)
    print(f' Epoch {epoch}: Train loss: {train_loss}')
    print('Accuracy : ',accuracy)
    print('\n')
    nile.write(f' Epoch {epoch}\nTrain loss: {train_loss}\n')
    nile.write(f'Accuracy : {accuracy}')
    nile.write(f'\n\n')
    if epoch % 5 == 0 :
        model_name = f'{epoch}model.pt'
        PATH = PATH + model_name
        torch.save(model.state_dict(),PATH)
nile.close()
# architecture 
# loading pickle file and predict

