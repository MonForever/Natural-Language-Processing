#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# In[4]:


df_movie = pd.read_csv('tmdb_5000_movies.csv')


# In[3]:


df_movie.head()


# In[5]:


x=df_movie.iloc[0]
x


# In[6]:


x["genres"]


# In[7]:


j = json.loads(x["genres"])


# In[8]:


j


# In[19]:



' '.join(''.join(jj["name"].split())for jj in j)


# In[14]:


print(list1)


# In[17]:


print(j)


# In[21]:


def genres_and_keywords_to_string(row):
    genres = json.loads(row["genres"])
    ' '.join(''.join(jj["name"].split())for j in genres)
    
    keywords = json.loads(row["keywords"])
    ' '.join(''.join(jj["name"].split())for j in keywords)
    
    return "%s %s" %(genres,keywords)


# In[22]:


#create a new string representation of genres and keywords

df_movie["string"] = df_movie.apply(genres_and_keywords_to_string, axis=1)


# In[23]:


#creating a tfidf vectorizer object
tfidf = TfidfVectorizer(max_features=2000)


# In[24]:


X = tfidf.fit_transform(df_movie["string"])


# In[25]:


X


# In[27]:


#generating mapping to String by using title as index
movieidx = pd.Series(df_movie.index, index=df_movie["title"])
movieidx


# In[29]:


idx = movieidx["Newlyweds"]
idx


# In[30]:


query = X[idx]
query


# In[39]:


#Compute cosine similarity between query and every other entry in X
scores = cosine_similarity(query, X)
scores


# In[40]:


#flattening the 1*N matrix to a 1-D array
scores = scores.flatten()


# In[41]:


plt.plot("scores")


# In[42]:


(-scores).argsort()


# In[43]:


plt.plot(scores[(-scores).argsort()])


# In[45]:


#getting the top 5 matches
#ignoring the first one, as it would be the same movie itself
recommended_idx = (-scores).argsort()[1:6]


# In[47]:


df_movie["title"].iloc[recommended_idx]


# In[48]:


#create a function that generates recommendations
def movierecommendation(title):
    #get the row in the dataframe for the movie
    idx = movieidx[title]
    #calculate the oairwise similarities for this movie
    query = X[idx]
    scores = cosine_similarity(query, X)
    #flattening the 1*N matrix to a 1-D array
    scores = scores.flatten()
    recommended_idx = (-scores).argsort()[1:6]
    return df_movie["title"].iloc[recommended_idx]


# In[50]:


print('Recommendations for Movie:Newlyweds is')
print(movierecommendation('Newlyweds'))


# In[51]:


print('Recommendations for Movie:Runaway Bride is')
print(movierecommendation('Runaway Bride'))


# In[52]:


print('Recommendations for Movie:Avatar is')
print(movierecommendation('Avatar'))


# In[ ]:




