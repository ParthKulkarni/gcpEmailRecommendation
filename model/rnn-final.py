
# coding: utf-8

# In[106]:


import numpy as np
import pandas as pd
import pickle
import math
from pprint import pprint 

# In[107]:


TRAIN_PATH = '/home/niki/train2.pkl'


# In[108]:


# reading pickle
dtrain = pd.read_pickle(TRAIN_PATH)
tfidf = []
thread_ids = dtrain.thread_no.unique()


# In[109]:


# different dataframes for each thread & list of list is getting created
for id in thread_ids :
    d = dtrain[dtrain['thread_no'] == id]
    tfidf.append(d['bodyidx'])


# In[110]:


print(len(tfidf))


# In[111]:


#model parameters
hidden_size = 100
seq_length = 5
learning_rate = 1e-3
vector_length = 1000


# In[112]:

Wxh = np.random.randn(hidden_size, vector_length) * 0.01 #input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #input to hidden
Why = np.random.randn(vector_length, hidden_size) * 0.01 #input to hidden
bh = np.zeros((hidden_size, 1))
by = np.zeros((vector_length, 1))
hprev = np.zeros((hidden_size,1))

print('Wxh : ',Wxh.shape)
print('Whh : ',Whh.shape)
print('Why : ',Why.shape)
print('bh : ', bh.shape)
print('by : ', by.shape)


# In[115]:


def lossFun(inputs, targets, hprev):
    
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    inputs = inputs.transpose()
    targets = targets.transpose()
    # forward pass
    l = 0
    for t in range(len(inputs)):
        xs[t] = inputs[t]
        xs[t] = xs[t].reshape(1000,1)
        trg = targets[t].reshape(1000,1)

        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)                                                                                                            
        ys[t] = np.dot(Why, hs[t]) + by                                                                                                       

        diff = trg - ys[t]
        l += diff * diff
        
    ls = l.sum(axis = 0)
    ls = ls[0] / float(inputs.shape[0])
    ls = ls / 1000.0
    mse = (ls) / 2.0
    loss += mse
        
    # initalize vectors for gradient values for each set of weights 
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    
    # backward pass: compute gradients going backwards
    for t in reversed(range(len(inputs))):
        #derive our first gradient
        trg = targets[t].reshape(1000,1)
        dy = (-(trg - ys[t]) * ys[t]) / float(inputs.shape[0]) # backprop into y  
        #compute output gradient 
        dWhy += np.dot(dy, hs[t].T)
        #derivative of output bias
        dby += dy
        #backpropagate!
        dh = np.dot(Why.T, dy) + dhnext # backprop into h                                                                                                                                         
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity                                                                                                                     
        dbh += dhraw #derivative of hidden bias
        dWxh += np.dot(dhraw, xs[t].T) #derivative of input to hidden layer weight
        dWhh += np.dot(dhraw, hs[t-1].T) #derivative of hidden layer to hidden layer weight
        dhnext = np.dot(Whh.T, dhraw) 
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients                                                                                                                 
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


# In[117]:


num_epochs = 10
#for iter in range(num_epochs):
   # Wxh = np.random.randn(hidden_size, vector_length) * 0.01 #input to hidden
   # Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #input to hidden
   # Why = np.random.randn(vector_length, hidden_size) * 0.01 #input to hidden
   # bh = np.zeros((hidden_size, 1))
   # by = np.zeros((vector_length, 1))
   # hprev = np.zeros((hidden_size,1))

n = 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad                                                                                                                
smooth_loss = -np.log(1.0/vector_length)*seq_length # loss at iteration 0 
for iter in range(num_epochs):
    n = 0
    while n< len(tfidf):
        # checking mail length is greater than 2
        if(len(tfidf[n]) < 2) :
            n += 1
            continue

        length = len(tfidf[n])

        # conversion to numpy ndarray
        thread = tfidf[n]
        n += 1
        thread = thread.values
        flag = 0

        # reshaping numpy array to be in required input format
        for t in thread:
            if flag == 0:
                mails = t.reshape(1000,1)
                flag =1
                continue
            mails = np.hstack((mails,t.reshape(1000,1)))

        # creating input output pairs
        # forward seq_length characters through the net and fetch gradient 
        #print('Length : ',length)
        if(length > seq_length):
            s_len = 0
            f1 = 0
            while s_len < length:
                if (s_len + seq_length + 1 >= length):
                    f1 = 1
                    break
                inputs = mails[:,s_len:s_len + seq_length]
                targets = mails[:,s_len + 1 :s_len + seq_length + 1]
                #print(s_len,' ',s_len + seq_length)
                #print(s_len + 1,' ',s_len + seq_length + 1)
                s_len += seq_length
                loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
                smooth_loss = smooth_loss * 0.999 + loss * 0.001

#             if (f1 == 1):
            inputs = mails[:,length -seq_length - 1 :length-1]
            targets = mails[:,length -seq_length  :length]
            #print(length -seq_length - 1 ,' ',length-1)
            #print(length -seq_length,' ',length)
            #print('inputs')
            #print(inputs)
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

        else :
            inputs = mails[:,:-1]
            targets = mails[:,1:]
            #print(len(mails[0]))
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

    
        # perform parameter update with Adagrad                                                                                                                                                     
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                    [dWxh, dWhh, dWhy, dbh, dby],
                                    [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            # adagrad update 
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) 

    print ('iter %d, loss: %f' % (iter, smooth_loss)) # print progress
    iter += 1

    #     # move data pointer  
    #     p += seq_length 

