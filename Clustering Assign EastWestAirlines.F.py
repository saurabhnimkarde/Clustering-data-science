#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore")


# Perform clustering (hierarchical,K means clustering and DBSCAN) for the airlines data to obtain optimum number of clusters. 
# Draw the inferences from the clusters obtained.

# In[63]:


# Performing hierarchical clustering


# In[64]:


air = pd.read_csv("EastWestAirlines.csv")


# In[66]:


air.head()


# In[67]:


# Determining the number of rows and columns


# In[68]:


air.shape


# In[69]:


# performing business decisions


# In[70]:


air.iloc[:,1:].describe()


# In[71]:


# Correlation matrix


# In[72]:


air.iloc[:,1:].corr()


# In[73]:


# Determining null vales


# In[74]:


air.info()


# In[76]:


# Checking normality of the data


# In[75]:


sns.pairplot(air)


# In[77]:


array = air.values
array


# In[78]:


# Standardizing the data using StandardScaler


# In[79]:


scaler = StandardScaler().fit(array)
X = scaler.transform(array)


# In[80]:


X


# In[81]:


# Creating Dendrogram


# In[82]:


plt.figure(figsize = (15,5))
dendrogram = sch.dendrogram(sch.linkage(X,method = 'complete'))


# In[83]:


# Model building using agglomerative clustering


# In[84]:


hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'complete')


# In[85]:


y_hc = hc.fit_predict(X)


# In[86]:


cluster = pd.DataFrame(y_hc,columns = ['cluster'])


# In[87]:


cluster


# In[88]:


air['clust']=hc.labels_


# In[89]:


air


# In[90]:


air.iloc[:,1:].groupby(air.clust).mean()


# 

# In[91]:


# KMeans clustering


# In[92]:


data_air = pd.read_csv("EastWestAirlines.csv")


# In[93]:


data_air.head()


# In[94]:


array1 = data_air.values
array1


# In[95]:


scaler = StandardScaler().fit(array1)
X1 = scaler.transform(array1)

set_printoptions(precision = 2)
X1


# In[96]:


# creating Elbow curve or scree plot


# In[97]:


wcss =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("ELBOW CURVE")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")


# In[98]:


# Model building


# In[99]:


model = KMeans(n_clusters = 6)
model.fit(X1)


# In[100]:


c_4 = model.labels_
c_4


# In[101]:


md = pd.Series(model.labels_)
data_air['Clust'] = md


# In[102]:


data_air


# In[103]:


data_air.iloc[:,1:].groupby(data_air.Clust).mean()


# In[104]:


plt.figure(figsize=(10,7))
plt.scatter(data_air['Clust'], data_air['Balance'], c= c_4)


# 

# In[105]:


# Performing DBSCAN


# In[106]:


db_air = pd.read_csv("EastWestAirlines.csv")


# In[107]:


db_air.head()


# In[108]:


array2 = db_air.values
array2


# In[109]:


scale = StandardScaler().fit(array2)
X2 = scale.transform(array2)


# In[110]:


X2


# In[111]:


#building model


# In[112]:


dbscan = DBSCAN(eps = 1, min_samples = 13)
dbscan.fit(X2)


# In[113]:


#determining noisy points


# In[114]:


dbscan.labels_


# In[115]:


db_air['ClustM'] = dbscan.labels_
db_air


# In[116]:


pd.concat([db_air, cluster],axis=1)


# In[117]:


db_air.groupby('ClustM').agg(['mean']).reset_index()


# In[ ]:




