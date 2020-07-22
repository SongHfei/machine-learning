# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:40:14 2020

@author: 10094
"""

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import confusion_matrix

# from sklearn.datasets import make_blobs    # 生成样本点数据,生成聚类的数据
data = load_iris()
x = data.data
y = data.target
pca=PCA(n_components=2)
pca_data = pca.fit_transform(x)
x=pca_data
scaler = MinMaxScaler().fit(x.reshape(-1,1))
x = scaler.transform(x)
length = x.shape[0]
permutation = np.random.permutation(length)
x = x[permutation,]
y = y[permutation,]
k = 0.7
train_num = math.floor(length*k)
x_train = x[:train_num,]
y_train = y[:train_num,]
x_test = x[train_num:,]
y_test = y[train_num:,]

# kernel in ('linear', 'poly', 'rbf'):
model = svm.SVC(kernel='poly', gamma=2)                 # gamma=1/(2*σ^2)
model.fit(x_train, y_train) 
y_predict = model.predict(x_test)
ss = model.score(x_test,y_test)
print(ss)

def cm_plot(original_label, predict_label, pic=None):
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
    # plt.figure(figsize=(12,8))
    plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
            # annotate主要在图形中添加注释
            # 第一个参数添加注释
            # 第二个参数是注释的内容
            # xy设置箭头尖的坐标
            # horizontalalignment水平对齐
            # verticalalignment垂直对齐
            # 其余常用参数如下：
            # xytext设置注释内容显示的起始位置
            # arrowprops 用来设置箭头
            # facecolor 设置箭头的颜色
            # headlength 箭头的头的长度
            # headwidth 箭头的宽度
            # width 箭身的宽度
    
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.title('confusion matrix')
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.show()
# 开始画图部分
cm_plot(y_test,y_predict)

