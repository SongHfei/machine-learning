# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:51:42 2020

@author: 10094
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
from keras.datasets import boston_housing
# In[]参数设置
lr = 0.001 #学习率
Epochs = 100000 #训练次数
decay_step = 100 # 衰减步长
dacay_rate_value = 0.2 #衰减率 
lw = 4 #线宽

# In[]导入数据
(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()


x_train=train_data.reshape(-1,1)
x_test=test_data.reshape(-1,1)
# In[] 预处理
scaler=pre.MinMaxScaler().fit(x_train)
x_train=scaler.transform(x_train).reshape(-1,13)
y_train=train_targets.reshape(-1,1)
x_test=scaler.transform(x_test).reshape(-1,13)
y_test=test_targets.reshape(-1,1)

# In[] 定义输入输出
x = tf.placeholder(tf.float32, [None, 13])
y = tf.placeholder(tf.float32, [None, 1])

# In[] 建立graph
w1 = tf.Variable(tf.random_normal(shape=(13,32),mean=0.0,stddev=1.0,dtype=tf.float32))
b1 = tf.Variable(tf.zeros(shape=(1,32),dtype=tf.float32))
h1 = tf.matmul(x, w1) + b1
o1 = tf.nn.relu(h1)

w2 = tf.Variable(tf.random_normal(shape=(32,64),mean=0.0,stddev=1.0,dtype=tf.float32))
b2 = tf.Variable(tf.zeros(shape=(1,64),dtype=tf.float32))
h2 = tf.matmul(o1, w2) + b2
o2 = tf.nn.relu(h2)

w3 = tf.Variable(tf.random_normal(shape=(64,1),mean=0.0,stddev=1.0,dtype=tf.float32))
b3 = tf.Variable(tf.zeros(shape=(1,1),dtype=tf.float32))
o3 = tf.matmul(o2, w3) + b3
y_pre=tf.nn.leaky_relu(o3)
# y_pre = tf.nn.relu(o3)

# In[] 定义代价函数
loss = tf.reduce_mean(tf.square(y - y_pre))
# In[] 定义梯度下降
# train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
train_step = tf.train.exponential_decay(learning_rate=lr, global_step=Epochs, decay_steps=decay_step, decay_rate=dacay_rate_value)
# In[] 变量初始化
initializer = tf.global_variables_initializer()
# 创建Session
with tf.Session() as sess:
    sess.run(initializer)
    for epoch in range(Epochs):
        sess.run(train_step,
                 feed_dict = {x:x_train,y:y_train})
    pre_out = sess.run(y_pre,feed_dict = {x:x_test,y:y_test})
    plt.figure(figsize=(12,8))
    plt.plot(y_test,linewidth=lw)
    plt.plot(pre_out,linewidth=lw)
    plt.legend(('y_test','y_pre'))
    plt.show()
