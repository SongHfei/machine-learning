# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:01:22 2020

@author: 10094
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm                    # 直接用sklearn中的包
from sklearn.datasets import make_blobs    # 生成样本点数据,生成聚类的数据

# 随机生成两个数据簇，保证线性可分
X, y = make_blobs(n_samples=100, centers=2, random_state=3)      # 100个样本，两个簇
clf = svm.SVC(kernel='linear', C=1000.0)                          # 构建分类器
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s = 30, cmap=plt.cm.Paired)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 计算决策边界
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 会值决策边界以及间隔
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
plt.show()


