# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:04:21 2020

@author: 10094
"""
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from  sklearn.cluster import DBSCAN,KMeans
from sklearn import svm

# from sklearn.datasets import make_blobs    # 生成样本点数据,生成聚类的数据


data = load_iris()
x = data.data
y = data.target
pca=PCA(n_components=2)
pca_data = pca.fit_transform(x)



# dbscan = DBSCAN(eps=0.4, min_samples=4)
# dbscan.fit(pca_data) 
# label_pred = dbscan.labels_

kmeans = KMeans(n_clusters = 3,verbose= 1)
kmeans.fit(pca_data)
label_pred = kmeans.labels_

def show_result(data):
    x0 = data[label_pred == 0]
    x1 = data[label_pred == 1]
    x2 = data[label_pred == 2]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')  
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
    plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')  
    plt.xlabel('sepal length')  
    plt.ylabel('sepal width')  
    plt.legend(loc=2)  
    plt.show()   
show_result(pca_data)



