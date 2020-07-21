# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:50:05 2020

@author: 10094
"""


#加载波士顿房价数据集
import keras
from keras import *
import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import sklearn.preprocessing as pre
# (x_train,y_train),(x_test,y_test)=boston_housing.load_data()
Epochs=1000
Batch_size=10
#准备数据，标准化
data = np.load('boston_housing.npz')
x = data['x'].reshape(-1,1)
y = data['y']

scaler = pre.MinMaxScaler().fit(x)
x=scaler.transform(x)
x = x.reshape(-1,13)
x = np.hstack((x,x,x,x,x))
x = x[:,:55]

x_train = x[:404,]
y_train = y[:404,]
x_test = x[404:,]
y_test = y[404:,]





        
model=models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(x_train.shape[1],)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

history = model.fit(x_train,y_train,epochs=Epochs,batch_size=Batch_size,verbose=1)
loss = history.history['loss']
mas = history.history['mean_absolute_error']
plt.figure(figsize=(12,8))
plt.plot(loss,linewidth=3)
plt.plot(mas,linewidth=3)
plt.legend(('loss','mas'))
plt.show()
# print(history.history['loss'])
test_mse_score,test_mae_score=model.evaluate(x_test,y_test)
print(test_mae_score)

test_pre=model.predict(x_test)
plt.figure(figsize=(12,8))
plt.plot(y_test,linewidth=3)
plt.plot(test_pre,linewidth=3)
plt.legend(('test_targets','test_pre'))
plt.show()
 
 
 