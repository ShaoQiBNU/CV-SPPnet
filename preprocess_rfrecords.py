#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess flower data: 
           train:
           1. crop data, size exchange into 500 x 500
           2. resize data, size exchange into 400 x 400, 300 x 300, 250 x 250
           3. save data and label into TFRecords

           test:
           save test data and label into TFRecords

读取原始数据，

将train数据集裁剪成500 x 500，然后最邻近重采样成 400 x 400, 300 x 300, 250 x 250，保存成TFRecords；
将test数据保存成TFRecords。

@author: shaoqi
"""

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
import PIL
from collections import defaultdict
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

##################### load flower data ##########################
def flower_preprocess(flower_folder, flower_crop, resize_list):

	'''
	flower_floder: flower original path 原始花的路径
	flower_crop: 处理后的flower存放路径
	'''

	######## flower dataset label 数据label ########
	labels = scipy.io.loadmat('/Users/shaoqi/Desktop/SPP/data/imagelabels.mat')
	labels = np.array(labels['labels'][0])-1


	######## flower dataset: train test valid 数据id标识 ########
	setid = scipy.io.loadmat('/Users/shaoqi/Desktop/SPP/data/setid.mat')
	test = np.array(setid['trnid'][0]) - 1
	np.random.shuffle(test)
	train = np.array(setid['tstid'][0]) - 1
	np.random.shuffle(train)


	######## flower data TFRecords save path TFRecords保存路径 ########
	writer_500 = tf.python_io.TFRecordWriter("/Users/shaoqi/Desktop/SPP/data/tfrecords/flower_train_500.tfrecords") 
	writer_400 = tf.python_io.TFRecordWriter("/Users/shaoqi/Desktop/SPP/data/tfrecords/flower_train_400.tfrecords") 
	writer_300 = tf.python_io.TFRecordWriter("/Users/shaoqi/Desktop/SPP/data/tfrecords/flower_train_300.tfrecords") 
	writer_250 = tf.python_io.TFRecordWriter("/Users/shaoqi/Desktop/SPP/data/tfrecords/flower_train_250.tfrecords") 
	writer_test = tf.python_io.TFRecordWriter("/Users/shaoqi/Desktop/SPP/data/tfrecords/flower_test.tfrecords")


	######## flower data path 数据保存路径 ########
	flower_dir = list()

	######## flower data dirs 生成保存数据的绝对路径和名称 ########
	for img in os.listdir(flower_folder):
        
        ######## flower data ########
		flower_dir.append(os.path.join(flower_folder, img))

	######## flower data dirs sort 数据的绝对路径和名称排序 从小到大 ########
	flower_dir.sort()


	###################### flower train data ##################### 
	for tid in train:
		######## open image and get label ########
		img=Image.open(flower_dir[tid])
		
		######## get width and height ########
		width,height=img.size

		######## crop paramater ########
		h=500
		x=int((width-h)/2)
		y=int((height-h)/2)


		################### crop image 500 x 500 and save image ##################
		img_crop=img.crop([x,y,x+h,y+h])

		######## img to bytes 将图片转化为二进制格式 ########
		img_500=img_crop.tobytes()

		######## build features 建立包含多个Features 的 Example ########
		example_500 = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[tid]])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_500])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[500])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[500]))
            }))

		######## 序列化为字符串,写入到硬盘 ########
		writer_500.write(example_500.SerializeToString())


		################# resize image and save 400 x 400 ##################
		img_400=img_crop.resize((400,400),Image.NEAREST)

		######## img to bytes 将图片转化为二进制格式 ########
		img_400=img_400.tobytes()

		######## build features 建立包含多个Features 的 Example ########
		example_400 = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[tid]])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_400])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[400])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[400]))}))

		######## 序列化为字符串,写入到硬盘 ########
		writer_400.write(example_400.SerializeToString())


		################ resize image and save 300 x 300 ##################
		img_300=img_crop.resize((300,300),Image.NEAREST)

		######## img to bytes 将图片转化为二进制格式 ########
		img_300=img_300.tobytes()

		######## build features 建立包含多个Features 的 Example ########
		example_300 = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[tid]])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_300])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[300])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[300]))}))

		######## 序列化为字符串,写入到硬盘 ########
		writer_300.write(example_300.SerializeToString())


		################ resize image and save 250 x 250 ##################
		img_250=img_crop.resize((250,250),Image.NEAREST)

		######## img to bytes 将图片转化为二进制格式 ########
		img_250=img_250.tobytes()

		######## build features 建立包含多个Features 的 Example ########
		example_250 = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[tid]])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_250])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[250])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[250]))}))

		######## 序列化为字符串,写入到硬盘 ########
		writer_250.write(example_250.SerializeToString())


	##################### flower test data ####################	
	for tsd in np.sort(test):
        
        ####### open image and get width and height #######
		img=Image.open(flower_dir[tsd])
		width,height=img.size

		######## img to bytes 将图片转化为二进制格式 ########
		img=img.tobytes()

		######## build features 建立包含多个Features 的 Example ########
		example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[tsd]])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))}))

		######## 序列化为字符串,写入到硬盘 ########
		writer_test.write(example.SerializeToString())


################ main函数入口 ##################
if __name__ == '__main__':

	######### flower path 鲜花数据存放路径 ########
	flower_folder = '/Users/shaoqi/Desktop/SPP/data/102flowers'
	flower_crop='/Users/shaoqi/Desktop/SPP/data/flower_'

	######## resize paramater 重采样参数设定 ########
	resize_list=[500,400,300,250]
    
    ######## 数据预处理 ########
	flower_preprocess(flower_folder, flower_crop, resize_list)