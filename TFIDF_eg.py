
# coding: utf-8

# In[2]:


docA = "the cat sat on my face"


# In[3]:


docB = "the dog sat on my bed"


# In[4]:


bowA = docA.split(" ")


# In[5]:


bowA


# In[6]:


bowB = docB.split(" ")


# In[7]:


bowB


# In[8]:


wordSet = set(bowA).union(set(bowB))


# In[9]:


wordSet


# In[10]:


wordDictA = dict.fromkeys(wordSet,0)


# In[11]:


wordDictB = dict.fromkeys(wordSet,0)


# In[12]:


wordDictA


# In[13]:


for word in bowA:
    wordDictA[word]+=1
for word in bowB:
    wordDictB[word]+=1

wordDictA
# In[14]:


import pandas as pd


# In[15]:


pd.DataFrame([wordDictA,wordDictB])


# In[16]:


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.iteritems():
        tfDict[word]=count/float(bowCount)
    return tfDict


# In[17]:


tfBowA = computeTF(wordDictA, bowA)
tfBowB = computeTF(wordDictB, bowB)


# In[24]:


def computeIDF(docList):
    import math
    idfDict = {}
    N= len(docList)
    idfDict = dict.fromkeys(docList[0].keys(),0)
    for doc in docList:
        for word, val in doc.iteritems():
            if val > 0:
                idfDict[word] +=1
    for word, val in idfDict.iteritems():
        idfDict[word]= math.log(N/float(val))
    return idfDict


# In[25]:


idfs = computeIDF([wordDictA, wordDictB])


# In[26]:


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.iteritems():
        tfidf[word] = val * idfs[word]
    return tfidf


# In[27]:


tfidfBowA = computeTFIDF(tfBowA, idfs)
tfidfBowB = computeTFIDF(tfBowB, idfs)


# In[28]:


import pandas as pd
pd.DataFrame([tfidfBowA, tfidfBowB])

