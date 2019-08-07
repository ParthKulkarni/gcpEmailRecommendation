
# coding: utf-8

# In[34]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

new_mat=np.load("mymatrix1.dat")
new_mat = new_mat.transpose()
print(new_mat)
# G=nx.read_adjlist("mymatrix.dat", create_using=nx.DiGraph())
G=nx.from_numpy_matrix(new_mat,parallel_edges=False,create_using=nx.DiGraph())
G.edges(data=True)


# In[35]:


print(G)
labels = nx.get_edge_attributes(G,'weight')
print(labels)


pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos,edge_labels=labels)
# pos=nx.get_node_attributes(G)
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.show()

from networkx.drawing.nx_agraph import to_agraph 
A = to_agraph(G) 
A.layout('dot')                                                                 
A.draw('multi3.png')


# In[36]:


sorted(nx.weakly_connected_components(G))


# In[37]:


sorted(nx.strongly_connected_components(G))


# In[ ]:


#DEGREE CENTRALITY


# In[38]:


deg_cent = nx.in_degree_centrality(G)
# print(deg_cent)
# print(type(deg_cent))

import operator
sorted_x = sorted(deg_cent.items(), key=operator.itemgetter(1),reverse=True)
print(sorted_x)


# In[ ]:


#CLOSENESS CENTRALITY


# In[39]:


close_cent = nx.closeness_centrality(G)
print(close_cent)
print(type(close_cent))

import operator
sorted_x1 = sorted(close_cent.items(), key=operator.itemgetter(1),reverse=True)
print(sorted_x1)


# In[ ]:


#BETWEENNESS CENTRALITY


# In[31]:


betw_cent = nx.betweenness_centrality(G,normalized=True,endpoints=False)
print(betw_cent)
print(type(betw_cent))

import operator
sorted_x2 = sorted(betw_cent.items(), key=operator.itemgetter(1),reverse=True)
print(sorted_x2)


# In[50]:


pg = nx.pagerank(G,alpha=0.8)
# print(pg)
# print(type(pg))

import operator
from pprint import pprint
sorted_x3 = sorted(pg.items(), key=operator.itemgetter(1),reverse=True)
pprint(sorted_x3)

