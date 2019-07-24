#!/usr/bin/env python
# coding: utf-8

# # 1. Create V, C

# In[1]:


import nltk
import numpy as np
from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize
from math import log


# In[2]:


nltk.download('brown')
nltk.download('stopwords')
nltk.download('punkt')


# In[3]:


words = ' '.join(brown.words())
print (words[:200])


# In[4]:


from collections import defaultdict, Counter
stop = set(stopwords.words('english'))
wordsNew = [word for word in word_tokenize(words.lower()) if word not in stop and word.isalpha()]
wordsNew = np.asarray(wordsNew)


# In[5]:


len(wordsNew)


# In[6]:


wordsCount = Counter(wordsNew)
V = np.asarray([e[0] for e in wordsCount.most_common(5000)])
C = V[:2000]


# # 2. Calculate P(c) and P(c|w)

# In[7]:


def word2map(words):
    word2idx = defaultdict(int);
    for word in words:
        word2idx[word] = len(word2idx)
    return word2idx


# In[8]:


V2idx = word2map(V);
C2idx = word2map(C);


# In[9]:


V_SIZE = len(V)
C_SIZE = len(C)
HALF_WINDOW = 2
M = len(wordsNew)

# prepare for the calculation of Pr(c) and Pr(c|w)
# use ones to apply laplace smoothing
print ("counting context appearance...")
window_count = np.ones((V_SIZE, C_SIZE))
core_count = np.ones((1, C_SIZE))
for i in range(M):
    w = wordsNew[i]
    if not w in V2idx:
        continue
    else:
        wid = V2idx[w]
    for j in range(i - HALF_WINDOW, i + HALF_WINDOW + 1):
        if j < 0 or j >= M or j == i:
            continue
        c = wordsNew[j]
        if not c in C2idx:
            continue
        else:
            cid = C2idx[c]
        window_count[wid][cid] += 1
        core_count[0][cid] += 1
print('done')


# In[10]:


# calculate Pr(c) and Pr(c|w)
print ("calculating probability...")
pwc, pc = window_count, core_count
for i in range(len(pwc)):
    pwc[i] = pwc[i] / pwc[i].sum()
pc = pc / pc.sum()

# calculate pointwise mutual information
r = np.zeros((V_SIZE, C_SIZE))
for i in range(V_SIZE):
    for j in range(C_SIZE):
        r[i][j] = max(0, log(pwc[i][j] /  pc[0][j]))

# save representation matrix to file
print ("saving representation...")
np.save("representation-" + str(C_SIZE) + ".npy", r)
print('done')


# # 3. Dimension Reduction

# In[12]:


from sklearn.decomposition import PCA
U_SIZE = 100
r = np.load("representation-" + str(C_SIZE) + ".npy");
pca = PCA(n_components = U_SIZE);
cr = pca.fit_transform(r);
np.save("representation-" + str(U_SIZE) + ".npy", cr);


# # 4. Clustering

# In[13]:


import random
from numpy.linalg import norm

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from queue import PriorityQueue


# In[14]:


def map2word(words):
    idx2word = defaultdict(str);
    for i, word in enumerate(words):
        idx2word[i] = word
    return idx2word


# In[15]:


idx2V = map2word(V)
idx2C = map2word(C)


# In[16]:


CLUSTER_NUMBER = 100
CLUSTER_THRESHOLD = 20
# cluster words
X = np.load("representation-" + str(U_SIZE) + ".npy")
kmeans = KMeans(n_clusters = CLUSTER_NUMBER).fit(X)

# sort words by distance to center
word_groups = {i:PriorityQueue() for i in range(CLUSTER_NUMBER)}
for i in range(len(X)):
    representation = X[i]
    word = idx2V[i]
    center_id = kmeans.predict(X[i].reshape(1, -1))[0]
    dist = norm(representation - kmeans.cluster_centers_[center_id])
    word_groups[center_id].put((float(dist), word))

# print only relatively large groups
for i in range(CLUSTER_NUMBER):
    if word_groups[i].qsize() < CLUSTER_THRESHOLD:
        continue
    count = 0
    for j in range(CLUSTER_THRESHOLD):
        if word_groups[i].empty():
            break
        print (word_groups[i].get()[1],)
        count += 1
        if count % 10 == 0:
            print
    print ("\n******************************")


# # 5. Nearest Neighbor

# In[17]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


# use cosine distance
tr = 1 - cosine_similarity(np.load("representation-" + str(U_SIZE) + ".npy"))
np.fill_diagonal(tr, 0)

neigh = KNeighborsClassifier(n_neighbors = 2, metric = "precomputed")
neigh.fit(tr, np.zeros((V_SIZE)))

# find NN for 25 random words
rand_inds = random.sample([i for i in range(V_SIZE)], 25)
for i in rand_inds:
    w = idx2V[i]
    dist, ind = neigh.kneighbors(tr[i].reshape(1, -1))
    uid = ind[0][1]
    u = idx2V[uid]
    print ("%-15s %-15s %f" %(w, u, dist[0][1]))


# In[ ]:




