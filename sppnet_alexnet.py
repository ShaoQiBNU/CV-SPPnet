# -*- coding: utf-8 -*-
'''
author: Shao Qi

sppnet_alexnet: 将SPP空间金字塔池化结构嫁接于Alexnet模型上，实现鲜花数据的分类
'''

##################### load packages #####################
import numpy as np
import os
import sys
import tarfile
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image
import math
import random
import re
import scipy.io
from skimage import io, data
import PIL
from collections import defaultdict
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

########## set net parameters ##########
#### 102 classes ####
n_classes=102

#### epochs ####
epochs=1

#### learning rate ####
learning_rate=0.00001

#### dropout probability
dropout=0.5

#### batch size ####
batch_size=102

#### spatial pool size ####
spatial_pool_size=[4, 2, 1]
spatial_pool_dim=sum([i*i for i in spatial_pool_size])

######### flower path train and test ########
flower_folder_train = ['flower_train_250.tfrecords','flower_train_300.tfrecords','flower_train_400.tfrecords','flower_train_500.tfrecords']
flower_folder_test=['flower_test.tfrecords']

######### flower size ########
flower_size=[250,300,400,500]

############### get flower data train ###############
def flower_batch(filename, batch_size, h):

    '''
    filename: TFRecord路径
    '''

    ########### 根据文件名生成一个队列 ############
    filename_queue = tf.train.string_input_producer([filename])

    ########### 生成 TFRecord 读取器 ############
    reader = tf.TFRecordReader()
    
    ########### 返回文件名和文件 ############
    _, serialized_example = reader.read(filename_queue)

    ########### 取出example里的features #############
    features = tf.parse_single_example(serialized_example,
      features={
      'label': tf.FixedLenFeature([], tf.int64),
      'img' : tf.FixedLenFeature([], tf.string),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64)})
    
    ########### 将序列化的img转为uint8的tensor #############
    img = tf.decode_raw(features['img'], tf.uint8)

    ########### 将label转为int32的tensor #############
    label = tf.cast(features['label'], tf.int32)
    
    ########### 将图片调整成正确的尺寸 ###########
    img = tf.reshape(img, [h, h, 3])
    img = tf.cast(img, tf.float32) * (1. / 255)

    ########### 批量输出图片, 使用shuffle_batch可以有效地随机从训练数据中抽出batch_size个数据样本 ###########
    ##### shuffle batch之前，必须提前定义影像的size，size不可以是tensor，必须是明确的数字 ######
    ##### num_threads 表示可以选择用几个线程同时读取 #####
    ##### min_after_dequeue 表示读取一次之后队列至少需要剩下的样例数目 #####
    ##### capacity 表示队列的容量 #####
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity= 100, num_threads= 2, min_after_dequeue= 10)

    return img_batch, label_batch


def dense_to_one_hot(labels_dense, num_classes):
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

########## set net parameters ##########
def weight_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

########## set net parameters ##########
weights={
    'wc1':weight_var('wc1',[11,11,3,96]),
    'wc2':weight_var('wc2',[5,5,96,256]),
    'wc3':weight_var('wc3',[3,3,256,384]),
    'wc4':weight_var('wc4',[3,3,384,384]),
    'wc5':weight_var('wc5',[3,3,384,256]),
    'wd1':weight_var('wd1',[spatial_pool_dim*256,4096]),
    'wd2':weight_var('wd2',[4096,4096]),
    'out_w':weight_var('out_w',[4096,n_classes])
}

biases={

    'bc1': bias_var('bc1',[96]),
    'bc2': bias_var('bc2',[256]),
    'bc3': bias_var('bc3',[384]),
    'bc4': bias_var('bc4',[384]),
    'bc5': bias_var('bc5',[256]),
    'bd1': bias_var('bd1',[4096]),
    'bd2': bias_var('bd2',[4096]),
    'out_b': bias_var('out_b',[n_classes])
}


##################### build net model ##########################
########## define conv process ##########
def conv2d(name,x,W,b,strides=1,padding='SAME'):
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding=padding)
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x,name=name)

########## define pool process ##########
def maxpool2d(name,x,ksize=1,strides=1,padding='SAME'):
    return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,strides,strides,1],padding=padding,name=name)

########## define norm process ##########
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.0001, beta=0.75, name=name)

########## define net structure ##########
def Alexnet_spatial_pool(x, weights, biases, dropout):

    #### 1 conv ####
    ## conv ##
    conv1=conv2d('conv1', x, weights['wc1'], biases['bc1'], strides=4, padding='VALID')
    ## pool ##
    pool1=maxpool2d('pool1', conv1, ksize=3, strides=2, padding='VALID')
    ## norm ##
    norm1=norm('norm1', pool1, lsize=4)

    #### 2 conv ####
    ## conv ##
    conv2=conv2d('conv2', norm1, weights['wc2'], biases['bc2'], 1, padding='SAME')
    ## pool ##
    pool2=maxpool2d('pool2', conv2, ksize=3, strides=2, padding='VALID')
    ## norm ##
    norm2=norm('norm2', pool2, lsize=4)

    #### 3 conv ####
    ## conv ##
    conv3=conv2d('conv3', norm2, weights['wc3'], biases['bc3'], 1)

    #### 4 conv ####
    ## conv ##
    conv4=conv2d('conv4', conv3, weights['wc4'], biases['bc4'], 1)

    #### 5 conv ####
    ## conv ##
    conv5=conv2d('conv5', conv4, weights['wc5'], biases['bc5'], 1)
    
    #### spatial pool ####
    spatial_pool=Sppnet(conv5, spatial_pool_size)

    #### 1 fc ####
    fc1=tf.reshape(spatial_pool,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1=tf.nn.relu(fc1)

    ## dropout ##
    fc1=tf.nn.dropout(fc1, dropout)

    #### 2 fc ####
    #fc2=tf.reshape(fc1,[-1,weights['wd2'].get_shape().as_list()[0]])
    fc2=tf.add(tf.matmul(fc1,weights['wd2']),biases['bd2'])
    fc2=tf.nn.relu(fc2)

    ## dropout ##
    fc2=tf.nn.dropout(fc2, dropout)

    #### output ####
    out=tf.add(tf.matmul(fc2,weights['out_w']),biases['out_b'])
    
    return out

####################### spatial pool #####################
def Sppnet(conv5, spatial_pool_size):
    
    ############### get feature size ##############
    height=int(conv5.get_shape()[1])
    width=int(conv5.get_shape()[2])
    
    ############### get batch size ##############
    batch_num=int(conv5.get_shape()[0])

    for i in range(len(spatial_pool_size)):
        
        ############### stride ############## 
        stride_h=int(np.ceil(height/spatial_pool_size[i]))
        stride_w=int(np.ceil(width/spatial_pool_size[i]))
        
        ############### kernel ##############
        window_w=int(np.ceil(width/spatial_pool_size[i]))
        window_h=int(np.ceil(height/spatial_pool_size[i]))
        
        ############### max pool ##############
        max_pool=tf.nn.max_pool(conv5, ksize=[1, window_h, window_w, 1], strides=[1, stride_h, stride_w, 1],padding='SAME')

        if i==0:
            spp=tf.reshape(max_pool, [batch_num, -1])
        else:
            ############### concat each pool result ##############
            spp=tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [batch_num, -1])])
    
    return spp

################## get train and test data label ###################

saver = tf.train.Saver()

############## get batch train images ################
for epoch in range(epochs):
    index=np.arange(4)
    np.random.shuffle(index)

    for i in index:

        x_train, y_train = flower_batch(flower_folder_train[i], batch_size, flower_size[i])

        x=tf.placeholder(tf.float32,shape=x_train.get_shape())
        y=tf.placeholder(tf.int32,[batch_size,n_classes])

        pred=Alexnet_spatial_pool(x, weights, biases, dropout)
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        with tf.Session() as sess:
            init=tf.global_variables_initializer()
            sess.run(init)
            ########## 启动队列线程 ##########
            coord=tf.train.Coordinator()
            threads= tf.train.start_queue_runners(sess=sess, coord=coord)
            
            if os.path.exists('./alex_model_spp.ckpt'):
                saver.restore(sess, './alex_model_spp.ckpt')
            for j in range(10000):

                x_train_batch, y_train_batch = sess.run([x_train, y_train])

                y_train_batch=np.reshape(y_train_batch,[batch_size,1])
                print(y_train_batch)
                y_train_batch=dense_to_one_hot(y_train_batch, n_classes)

                for k in range(10):
                    sess.run(optimizer,feed_dict={x: x_train_batch, y: y_train_batch})
                    loss,acc=sess.run([ cost, accuracy],feed_dict={x: x_train_batch, y: y_train_batch})
                    print(j, k, loss, acc)    
                
                pre=sess.run(tf.argmax(pred,1), feed_dict={x: x_train_batch})
                print(pre)  

            saver.save(sess, './alex_model_spp.ckpt')
            coord.request_stop()
            coord.join(threads)
        sess.close()
        del sess


x_test, y_test = flower_batch(flower_folder_test, batch_size=1)

result=[]
labels=[]
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    Session.run(init)
    ########## 启动队列线程 ##########
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(sess=sess, coord=coord)
    saver.restore(sess, './alex_model_spp.ckpt')

    x_test_batch, y_test_batch = sess.run([x_test, y_test])

    x = tf.placeholder('float', shape = x_test.get_shape())
    y = tf.placeholder(tf.int32,[1,n_classes])

    y_test_batch=np.reshape(y_test,[1,1])
    labels.append(y_test_batch)

    pred_test=sess.run([tf.argmax(pred)],feed_dict={x: x_test_batch})
    result.append(tf.argmax(pred_test,1))
    print("predict test result:", pred_test, y_test_batch)

print("Test accuracy:", (sum(np.array(result) == np.array(labels)).astype('float')/len(labels)))