# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:34:28 2020

@author: Nandini senghani
"""

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
##########################################Airlines dataset###########################
Airline=pd.read_csv("EastWestAirlines.csv")
Airline.head()
Airline.describe()
Airline.isnull().sum()
Airline.isna().sum()
Airline.columns
#eda
Q1=Airline.quantile(0.25)
Q3=Airline.quantile(0.75)
IQR=Q3-Q1
A=(Airline<(Q1-1.5 * IQR))|(Airline>(Q3+1.5*IQR))#outliers are present confirmed
A
Airline_out= Airline[~((Airline<(Q1-1.5 * IQR))|(Airline>(Q3+1.5*IQR))).any(axis=1)]
Airline_out.shape
#we need to check with the client about the outliers and what could cause them .As there are huge number of outliers and replacing with median value of the column might affect the accuracy of the model.
#Normalization function
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
airline_norm = norm_func(Airline)
airline_norm.describe()
type(airline_norm)
k = list(range(7,20))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airline_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(airline_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,airline_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
model1=KMeans(n_clusters=15) 
model1.fit(airline_norm)

model1.labels_ 
md=pd.Series(model1.labels_)  
Airline['clust']=md 
airline_norm.head()
Airline = Airline.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
Airline.groupby(Airline.clust).mean()
 ######################################crime Dataset###################################
crime=pd.read_csv("crime_data.csv")
crime.head()
crime.describe()
crime.isnull().sum()
crime.isna().sum()
crime.columns
plt.boxplot(crime.Murder,1,"ro",1)
plt.boxplot(crime.Assault,1,"ro",1)
plt.boxplot(crime.UrbanPop,1,"ro",1)
plt.boxplot(crime.Rape,1,"ro",1)
crime.Rape.quantile(0.50) # 20.1
crime.Rape.quantile(0.95) # 39.745
crime.Rape = np.where(crime.Rape > 40 , 20.1 , crime.Rape)
#outliers removed
#Normalization function
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
crime_norm = norm_func(crime.iloc[:,1:])
crime_norm.describe()
type(crime_norm)
k1 = list(range(2,15))
k1
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k1:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(crime_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,crime_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
# Scree plot 
plt.plot(k1,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k1)
model2=KMeans(n_clusters=3) 
model2.fit(crime_norm)
model2.labels_ 
cluster_labels=pd.Series(model2.labels_)
crime['clust']=cluster_labels   
crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.groupby(crime.clust).mean()
crime.to_csv("Crime.csv",index=False)
