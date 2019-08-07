from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re
import json
 
class Summarizer: 

    def __init__(self, subject, input_text):
        self.subject = subject
        self.input_text = input_text
        self.top_n = 10
        self.n = 0
        # print(self.input_text)


    def read_article(self):
        
        sentences = []
        for body in self.input_text:
            body = re.sub('\n+','\n',body)
            #self.input_text = re.sub('\n',' ',self.input_text)
            body = re.sub('\t',' ',body)
            body = re.sub(' +',' ',body)
            strings = body.splitlines()
            body = ''
            for st in strings:
                st = st.strip()
                if len(st)>0:
                    #print(st[0])
                    if st[0]=='>':
                        continue
                    else:
                        body += ' ' + st
            article = body.split(". ")
            #print(article)
            

            for sentence in article:
                #print(sentence)
                
                sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        #sentences.pop() 
        #print(sentences)
        self.n = min(len(sentences), self.top_n)
        if self.n < 10:
            self.n = 5
        return sentences

    def sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []
     
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
     
        all_words = list(set(sent1 + sent2))
     
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
     
        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1
     
        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1
     
        return 1 - cosine_distance(vector1, vector2)
     
    def build_similarity_matrix(self, sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
     
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix


    def generate_summary(self):
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences =  self.read_article()

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = self.build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        #print(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

        for i in range(self.n):
          summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize texr
        data = {"subject":self.subject, "text": ". ".join(summarize_text)}
        json_data = json.dumps(data)
        return json_data

