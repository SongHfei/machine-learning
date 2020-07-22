# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:14:48 2020

@author: 10094
"""
import numpy as np
import keras
import math
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
from keras.models import *
from keras.layers import *
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD ,RMSprop, Adam
from sklearn.cluster import KMeans
np.random.seed(0)
# In[] load data and data process
data = np.load('array_sum.npy')
train_num = 50
error_ave = []
# 1打乱数据 预测平均误差为: 0.032979106556479884
# permutation = np.random.permutation(data.shape[1])
# data = data[:,permutation]

# 2选取z最大点：预测平均误差为: 3830339094997.4424
data = np.sort(data)
data = data[:,::-1]

# 3聚类取出各部分数据  预测平均误差为: 83369014864212.34
# kmeans = KMeans(n_clusters = 5,verbose= 0)
# data_all = np.zeros(30897).reshape(-1,1)

# for i in range(4):
#     datai=data[i,].reshape(-1,1)
#     # data0 = np.zeros(1).reshape=(1,1)

#     kmeans.fit(datai)
#     label_pred = kmeans.labels_
#     x0 = datai[label_pred == 0]
#     x1 = datai[label_pred == 1]
#     x2 = datai[label_pred == 2]
#     x3 = datai[label_pred == 3]
#     x4 = datai[label_pred == 4]
#     data_x = np.vstack((x0[0:10],x1[0:10],x2[0:10],x3[0:10],x4[0:10]))
#     data_y = np.vstack((x0[10:,],x1[10:,],x2[10:,],x3[10:,],x4[10:,]))
#     data_single = np.vstack((data_x,data_y))
#     data_all = np.hstack((data_all,data_single))
# data = data_all.T
# data = data[1:5,]

# scaler = pre.StandardScaler().fit(data.reshape(-1,1))
# scaler_data = scaler.transform(data)
# data  = scaler_data
# x = data[:,0:train_num]
# y = data[:,train_num:]

scaler = pre.StandardScaler()
x = scaler.fit_transform(data[:,0:train_num])
y = scaler.fit_transform(data[:,train_num:])

for i in range(4):
    x_train = np.delete(x,i,axis=0)
    x_test = x[i,].reshape(1,-1)
    y_train = np.delete(y,i,axis=0)
    y_test = y[i,].reshape(1,-1)

# x_train = x[0:3,]
# x_test = x[3,].reshape(1,-1)
# y_train = y[0:3,]
# y_test = y[3,].reshape(1,-1)
    
    
    # In[] set value
    out_num = y_test.shape[1]
    hidden_layers = 15
    Epoch = 1000
    Batch_size = 1
    Validation_data = (x_test,y_test)
    
    # In[] build model
    model = Sequential()
    model.add(Dense(hidden_layers, input_shape=(train_num,)))
    model.add(Activation("relu"))
    # model.add(Dense(256,activation='sigmoid'))
    model.add(Dense(out_num))
    model.summary()
    # model.compile(loss="mape",optimizer=OPTIMIZER)
    
    model.compile(loss='mse',optimizer=Adam(lr=0.001,decay=1e-5))
    
    # model.compile(loss='mape', optimizer='SGD')
    # Callbacks = [keras.callbacks.ReduceLROnPlateau(mnitor='val_loss',
    #                                                factor=0.1,patience=5,mode='auto',)]
    
    history = model.fit(x_train, y_train, epochs=Epoch, batch_size=Batch_size,
                        # callbacks=Callbacks,
                        validation_data=Validation_data)
    loss = history.history['loss']
    mas = history.history['val_loss']
    
    plt.figure(figsize=(12,8))
    plt.plot(loss,linewidth=3)
    plt.plot(mas,linewidth=3)
    plt.legend(('loss','val_loss'))
    plt.show()
    print('loss:',loss[-1])
    
    # 预测
    pre = model.predict(x_test)
    pre = scaler.inverse_transform(pre)
    y_test = scaler.inverse_transform(y_test)
    mse = math.sqrt(mean_squared_error(pre, y_test))
    print('Test MSE: %.3f' % mse)
    
    index = []
    a = abs((y_test-pre)/y_test)
    b = np.isinf(a)
    for i in range(b[0].shape[0]):
        if b[0][i] == True:
            index.append(i)
    
    new_array = np.delete(a[0],index)
    jj=0
    for i in new_array:
        if i>=0.2:
            # print(i)
            jj+=1 
    pj = new_array.sum()/new_array.shape[0]
    print("预测绝对值百分比误差大与20的点个数为: %s" %jj)
    print("预测平均误差为: %s" %pj)
    error_ave.append(pj)