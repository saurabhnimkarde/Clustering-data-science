#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore")


# In[2]:


Crime_data = pd.read_csv("crime_data.csv")


# In[3]:


Crime_data.head()


# In[4]:


Crime_data.shape


# In[5]:


Crime_data.describe()


# In[6]:


Crime_data.info()  


# In[7]:


Crime_data.corr()


# In[8]:


plt.subplot(221)
sns.distplot(Crime_data['Murder'])
plt.subplot(222)
sns.distplot(Crime_data['Assault'])
plt.subplot(223)
sns.distplot(Crime_data['UrbanPop'])
plt.subplot(224)
sns.distplot(Crime_data['Rape'])


# In[9]:


plt.subplot(221)
sns.boxplot(Crime_data['Murder'])
plt.subplot(222)
sns.boxplot(Crime_data['Assault'])
plt.subplot(223)
sns.boxplot(Crime_data['UrbanPop'])
plt.subplot(224)
sns.boxplot(Crime_data['Rape'])


# In[10]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[11]:


data_norm = norm_func(Crime_data.iloc[:,1:])


# In[12]:


data_norm


# In[13]:


plt.figure(figsize=(15, 5))
dendrogram = sch.dendrogram(sch.linkage(data_norm,method = 'complete'))  


# In[14]:


plt.figure(figsize=(15, 5))
dendrogram = sch.dendrogram(sch.linkage(data_norm,method = 'single'))


# In[15]:


plt.figure(figsize=(15, 5))
dendrogram = sch.dendrogram(sch.linkage(data_norm,method = 'average'))


# In[17]:


plt.figure(figsize=(15, 5))
dendrogram = sch.dendrogram(sch.linkage(data_norm,method = 'centroid'))


# In[16]:


hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'complete')


# In[17]:


hc


# In[18]:


y_hc = hc.fit_predict(data_norm)
cluster = pd.DataFrame(y_hc,columns=['Clusters'])


# In[19]:


y_hc


# In[20]:


cluster


# In[21]:


Crime_data['Clusterid'] = hc.labels_


# In[22]:


Crime_data


# In[23]:


Crime_data.iloc[:,1:].groupby(Crime_data.Clusterid).mean()


# In[24]:


data = pd.read_csv("crime_data.csv")


# In[25]:


data.head()


# In[26]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[28]:


df_norm = norm_func(data.iloc[:,1:])
df_norm


# In[29]:


wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,random_state= 0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("ELBOW CURVE")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")


# In[30]:


model = KMeans(n_clusters = 4)
model.fit(df_norm)
model.labels_


# In[31]:


md = pd.Series(model.labels_)
data['clusterid']=md


# In[32]:


data


# In[33]:


data.iloc[:,1:].groupby(data.clusterid).mean()


# In[34]:


data2 = pd.read_csv("crime_data.csv")


# In[35]:


data2.head()


# In[36]:


data2_c = data2.iloc[:,1:]


# In[37]:


data2_c.head()


# In[38]:


array = data2_c.values
array


# In[39]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(array)
X = scaler.transform(array)


# In[40]:


X


# In[41]:


dbscan = DBSCAN(eps = 1, min_samples = 4)
dbscan.fit(X)


# In[42]:


dbscan.labels_


# In[43]:


cl = pd.DataFrame(dbscan.labels_,columns = ['Clusterid'])
cl


# In[44]:


pd.concat([data2, cl],axis = 1)


# In[ ]:




