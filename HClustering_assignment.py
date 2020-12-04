# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 19:26:07 2020

@author: Nandini senghani
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import	AgglomerativeClustering
########################################Airline dataset#############################
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
z = linkage(airline_norm, method="complete",metric="euclidean")
plt.figure(figsize=(30, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
h_Airline= AgglomerativeClustering(n_clusters=15,linkage='complete',affinity = "euclidean").fit(airline_norm) 
h_Airline.labels_
cluster_labels1=pd.Series(h_Airline.labels_)
Airline['clust']=cluster_labels1
Airline = Airline.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
Airline.groupby(Airline.clust).mean()
##########################################Crime_data dataset############################       
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
z1 = linkage(crime_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z1,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
h_crime= AgglomerativeClustering(n_clusters=3,linkage='complete',affinity = "euclidean").fit(crime_norm) 
h_crime.labels_
cluster_labels=pd.Series(h_crime.labels_)
crime['clust']=cluster_labels
crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.groupby(crime.clust).mean()
crime.to_csv("Crime.csv",index=False)








