# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:31:56 2020

@author: 10094
"""

import numpy as np
import matplotlib.pyplot as plt
import keras # 导入Keras
from keras.datasets import mnist # 从keras中导入mnist数据集
from keras.models import Sequential # 导入序贯模型
from keras.layers import Dense # 导入全连接层
from keras.optimizers import SGD # 导入优化函数
import sklearn.preprocessing as preprocessing
Epochs=100
batch_size=50
# (x_train, y_train), (x_test, y_test) = mnist.load_data() # 下载mnist数据集
data= np.load('mnist.npz')
x_train=data['x_train']
y_train=data['y_train']
x_test=data['x_test']
y_test=data['y_test']
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
# import matplotlib.pyplot as plt # 导入可视化的包

im = plt.imshow(x_train[0],cmap='gray')
plt.show()
print (y_train[0])


x_train = x_train.reshape(60000*784,1) # 将图片flatten，变成1D向量
x_test = x_test.reshape(10000*784,1) # 对测试集进行同样的处理
print(x_train.shape)
print(x_test.shape)
scaler=preprocessing.MinMaxScaler().fit(x_train)#归一化方式

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)


y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

def generator():
    while 1:
        row = np.random.randint(0,len(x_train),size=batch_size)
        # x = np.zeros((batch_size,x_train.shape[-1]))
        # y = np.zeros((batch_size,))
        x = x_train[row]
        y = y_train[row]
        yield x,y
# In[] 设置参数
        
model = Sequential() # 构建一个空的序贯模型
# 添加神经网络层
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()


model.compile(optimizer=SGD(),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=50,epochs=100,validation_data=(x_test,y_test)) # 此处直接将测试集用作了验证集
# history = model.fit_generator(generator(),epochs=Epochs,steps_per_epoch=len(x_train)//(batch_size*Epochs))

score = model.evaluate(x_test,y_test)
print("loss:",score[0])
print("accu:",score[1])
