
# coding: utf-8

# # Preprocessing function definition and reading into thread list

# In[107]:


import re
import sys
import glob
import string
from pprint import pprint
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim
from gensim import models

import os, re, operator, warnings
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
get_ipython().run_line_magic('matplotlib', 'inline')
import spacy
nlp = spacy.load("en")

folder_path = "/home/niki/Documents/BE_Project/gcpEmailRecommendation/Scraping/mini_deb/*"
sys.path.insert(0, '/home/niki/Documents/BE_Project/my_EmailRecommmendation/Preprocessing')


import preprocessing
import read_file
import datetime


def get_sender(msg):
    msg = email.message_from_string(msg)
    mfrom = msg['From'].split('<')[0]
    return mfrom

def extract_debian(text):
    text = text.split('\n\n\n')
    header = text[0].split('\n')
    body = text[1]
    sender = header[2].split(':')[1].split('<')[0]
#     print('Sender',sender)
#     print('Body \n',body)
    return sender,body

def clean_debian(temp):
    temp = temp.strip()
    temp = re.sub('\n+','\n',temp)
    temp = re.sub('\n',' ',temp)
    temp = re.sub('\t',' ',temp)
    temp = re.sub(' +',' ',temp)
    return temp

def deb_lemmatize(doc):        
    doc = nlp(doc)
    article, skl_texts = [],[]
    for w in doc:
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
            article.append(w.lemma_)
        if w.text == '\n':                
            skl_texts.append(article)
            article = []       
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

df = pd.DataFrame(columns=['body','replier', 'thread_no'])
users = []
folder = glob.glob(folder_path)
th_no = 0
obj = preprocessing.preprocess()
cnt = 0
count_file = 0
thread_list=[]
try:
    for fol in tqdm_notebook(folder):
        files = glob.glob(fol+'/*.txt')
        flag = 0
        t = ''
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


# In[108]:


text = []
t = ''
for thr in thread_list:
    for mail in thr:
        temp = ''
        sender = mail['From']
        temp   = mail['content']
        users.append(sender)
        temp = deb_toppostremoval(temp)
        temp = obj.replace_tokens(temp)
        temp = clean_debian(temp)
        if temp == '':
            cnt += 1
            continue
        t += temp + '\n'
print(cnt)
print(count_file)
t = t[:-2]
text = deb_lemmatize(t)


# # Topic Modelling

# In[109]:


bigram = gensim.models.Phrases(text)
text = [bigram[line] for line in text]

dictionary = Dictionary(text)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

corpus = [dictionary.doc2bow(txt) for txt in text]
tfidf = models.TfidfModel(corpus)
corpus = tfidf[corpus]


# In[110]:


ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
ldamodel.show_topics()

