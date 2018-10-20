SPP net详解
=============================

# 一. 背景

> SPP-Net是出自2015年发表在IEEE上的论文-《Spatial Pyramid Pooling in Deep ConvolutionalNetworks for Visual Recognition》。在此之前，所有的神经网络都是需要输入固定尺寸的图片，比如224x224（ImageNet）、32x32(LenNet)、96*96等。这样对于我们希望检测各种大小的图片的时候，需要经过crop，或者warp等一系列操作，这都在一定程度上导致图片信息的丢失和变形，限制了识别精确度。而且，从生理学角度出发，人眼看到一个图片时，大脑会首先认为这是一个整体，而不会进行crop和warp，所以更有可能的是，我们的大脑通过搜集一些浅层的信息，在更深层才识别出这些任意形状的目标。


> CNN主要由两部分组成，卷积部分和其后的全连接部分。卷积部分通过滑窗进行计算，并输出代表激活的空间排布的特征图（feature map），比如任意图片大小(w,h),任意的卷积核size（a,b），默认步长为１，我们都会得到卷积之后的特征图F(w-a+1,h-b+1)，所以这部分对图片大小没有要求；而全连接层的神经元设定之后是固定的（Input layer 神经元个数），每一个都对应一个特征，所以正是因为全连接层的存在，才导致CNN的输入必须是固定的。

# 二. SPP net详解

### 1. SPP net设计
> 为了解决CNN输入图像大小必须固定的问题，何凯明提出了SPP(Spatial Pyramid Poolling)——空间金字塔池化，从而可以使得输入图像的大小任意。何凯明在最后一个卷积层之后添加了SPP层，SPP层对features做pool，然后生成固定尺寸的输出，再feed进全连接层，然后输出。SPP的具体设计如图所示：

![image](https://github.com/ShaoQiBNU/SPPnet/blob/master/images/1.png)

> 论文中采用的是三层的金字塔池化，池化方式是maxpool，pyramid level设置为(4,2,1)。黑色图片代表卷积之后的特征图，接着以不同大小的块来提取特征，分别是4x4，2x2，1x1。4x4表示将特征图切分为4x4=16个小的特征图，如图左侧16个蓝色小格子的图；2x2表示将特征图切分为2x2=4个小的特征图，如图中间4个绿色小格子的图；1x1表示对整个特征图进行池化，如图右侧的灰色格子；其中256代表channels。这样一来就可以得到16+4+1=21种不同的块(Spatial bins)，每个块提取出一个特征，如maxpool，就是计算每个块的最大值，从而得到一个输出单元，最终得到一个21x256维特征的输出，然后进入全连接层。

### 2.bin pool size确定
> 此外，论文里还给出了bin size的确定方法：假设pyramid level size为nxn，最后一个卷积层的结果为axa，则bin pool的相关参数为：
> window size=ceil[a/n]，向上取整
> stride size=floor[a/n]，向下取整
>
> 例如，卷积之后的结果为13x13，pyramid level为(3,2,1)，则各个bin pool的参数如下：

![image](https://github.com/ShaoQiBNU/SPPnet/blob/master/images/2.png)

```
pool 3x3
window size = ceil[13/3] = 5
stride size = floor[13/3] = 4

pool 2x2
window size = ceil[13/3] = 7
stride size = floor[13/3] = 6

pool 1x1
window size = ceil[13/3] = 13
stride size = floor[13/3] = 13

三个bin pool之后输出的bins size为 3x3 + 2x2 + 1x1 = 14
```
> 但是这种pool的参数设定会有问题，例如假设conv之后为 7 x 7，pyramid level size为4x4，计算出bin pool size如下：

```
pool 4x4
window size = ceil[7/4] = 2
stride size = floor[7/4] = 1

理想状态最终输出bins size为 4x4，但是实际输出bins size为 6x6
```

> 对公式进行修订，如下：
window size=ceil[a/n]，向上取整
stride size=ceil[a/n]，向上取整
pool的padding方式为 SAME，这样便可以保证SPP层之后得到同样维度的特征输出。

### 3. 训练
> SPP net训练方式有两种：Single-size training和Multi-size training。

#### (1) Single-size training
> 采用单一尺寸的影像训练，SPP layer的bin pool按照2中的修订方式确定参数，做BP训练。

#### (2) Multi-size training
> 多尺度影像训练，即采用两个尺度进行训练：224x224和180x180，224x224的影像通过crop得到，而180x180通过224x224缩放得到，之后迭代训练，即用224训练一个epoch，然后用180训练一个epoch，交替进行。两种尺度下，SPP layer之后输出的特征维度均相同，参数共享，之后连接全连接层即可，这样训练的好处是可以更快的收敛。

# 三. SPP net应用于分类

> 作者将spp net嫁接在四个CNN网络上对ImageNet数据、VOC2017数据和Caltech101数据进行分类，并做了single-size和multi-size的对比，结果如下：

![image](https://github.com/ShaoQiBNU/SPPnet/blob/master/images/3.png)

![image](https://github.com/ShaoQiBNU/SPPnet/blob/master/images/4.png)

## (一) 数据预处理

> 数据集采用是102类的鲜花数据集，链接为：http://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/.， 数据格式为jpg，数据size不全都一样，因此首先对数据进行预处理。数据集里有setid.mat参数文件，此文件是一个字典数据，其中包含三个key：trnid，tstid，valid，tstid为训练集id，trnid为测试集id，valid为验证集id。将训练集中的影像全部做crop处理，裁剪成500 x 500大小的影像，然后对其做重采样，采样方式为最邻近插值，分别重采样成 400 x 400，300 x 300，250 x 250大小的影像，写成TFrecords格式存储到硬盘上。测试集影像单独取出，不做任何操作，写成TFrecords格式存储到硬盘上。代码如下：

```python
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
```

## (二) 模型

> 模型采用alexnet，第5个卷积层后添加SPP layer，pyramid level设置为[8, 6, 4]。代码如下：

```python
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
```

# 四. SPP net应用于目标检测

> SPP net应用于目标检测的流程如下：

```
1、使用 EdgeBoxes 算法生成候选区域

2、将全图feed进 CNN 网络提取全图特征

3、让候选区域与feature map直接映射，得到候选区域的映射特征向量(这是映射来的，不需要过CNN)

4、映射过来的特征向量大小不固定，所以这些特征向量塞给SPP层(空间金字塔变换层)，SPP层接收任何大小的输入，输出固定大小的特征向量，再feed进FC层

5、将FC层的输出特征输入到 SVM 分类器，判别输入类别

4、以回归的方式精修候选框
```

## (一) SPP net优势

> 与R-CNN对比，SPP net提出两种改进：

```
1、CNN网络后面接的FC层需要固定的输入大小，SPP net的SPP层克服了限制网络的输入大小的瓶颈

2、R-CNN 提取特征的顺序是先生成区域、再通过卷积神经网络提取特征，虽然相比传统的滑窗策略减少了大量的运算，但是依旧有大量的算力冗余。SPP net改变了一下顺序——先卷积，再在特征图上提取区域特征。
```

## (二) 候选区域到全图的特征映射

> SPPNet提出了一种从候选区域到全图的特征映射(feature map)之间的对应关系，通过此种映射关系可以直接获取到候选区域的特征向量，不需要重复使用CNN提取特征，从而大幅度缩短训练时间；具体过程解释如下。

### 1. 感受野及计算

> 在卷积神经网络中，感受野的定义是卷积神经网络每一层输出的特征图（feature map）上的像素点在原始图像上映射的区域大小。卷积神经网络CNN中，某一层输出结果中一个元素所对应的输入层的区域大小，被称作感受野receptive field。如图所示：

![image](https://github.com/ShaoQiBNU/SPPnet/blob/master/images/3.jpg)

> CNN中输入输出影像的大小满足如下关系：
```
隐藏层边长（输出的边长） = （W - K + 2P）/S + 1 

其中 W是输入特征的大小，K是卷积核大小，P是填充大小，S是步长（stride）

output field size = ( input field size - kernel size + 2*padding ) / stride + 1

output field size 是卷积层的输出，input field size 是卷积层的输入
```
> 因此当知道某个卷积层的大小需要反推上一个卷积层的大小时——感受野，只需将上面的公式进行变换：
```
 input field size = （output field size - 1）* stride - 2*padding + kernel size
```
> 所以，感受野的大小是由kernel size，stride，padding , outputsize 一起决定的。另外，在计算感受野时，需要注意以下几点：

```
（1）第一层卷积层的输出特征图像素的感受野的大小等于滤波器的大小

（2）深层卷积层的感受野大小和它之前所有层的滤波器大小和步长有关系

（3）计算感受野大小时，忽略了图像边缘的影响，即不考虑padding的大小
```

> 这里的每一个卷积层还有一个strides的概念，这个strides是之前所有层stride的乘积。即strides（i） = stride(1) * stride(2) * ...* stride(i-1) 。
>
> 关于感受野大小的计算采用top to down的方式， 即先计算最深层在前一层上的感受野，然后逐渐传递到第一层，使用公式可以表示如下：　

```python
RF = 1 #待计算的feature map上的感受野大小
for layer in (top layer To down layer):
    RF = ((RF -1)* stride) + fsize
   
stride 表示卷积的步长； fsize表示卷积层滤波器的大小
```

> 用python实现Alexnet和VGG16网络的每层输出feature map的感受野大小，代码如下：

```python
#!/usr/bin/env python

net_struct = {
'alexnet': {
'net':[[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0]], 
'name':['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5']},
       
'vgg16': {
'net':[[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[3,1,1],
[2,2,0],[3,1,1],[3,1,1],[3,1,1],[2,2,0],[3,1,1],[3,1,1],[3,1,1],[2,2,0]],
'name':['conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2',
'conv3_3', 'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5']}}

imsize = 224

def outFromIn(isz, net, layernum):
    totstride = 1
    insize = isz
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2*pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride

def inFromOut(net, layernum):
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF -1)* stride) + fsize
    return RF

if __name__ == '__main__':
    print("layer output sizes given image = %dx%d" % (imsize, imsize))
    
    for net in net_struct.keys():
        print('************net structrue name is %s**************'% net)
        for i in range(len(net_struct[net]['net'])):
            p = outFromIn(imsize,net_struct[net]['net'], i+1)
            rf = inFromOut(net_struct[net]['net'], i+1)
            print("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (net_struct[net]['name'][i], p[0], p[1], rf))
```

### 2. 感受野的特征映射

> 通常，我们需要知道网络里面任意两个feature map之间的坐标映射关系（一般是中心点之间的映射），如下图，我们想得到map 3上的点p3映射回map 2所在的位置p2（橙色框的中心点），计算公式如下：
> <a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}&space;&plus;\frac{k_{i}&space;-&space;1}{2}-&space;padding" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}&space;&plus;\frac{k_{i}&space;-&space;1}{2}-&space;padding" title="p_{i} = s_{i}\cdot p_{i+1} +\frac{k_{i} - 1}{2}- padding" /></a>

![image](https://github.com/ShaoQiBNU/SPPnet/blob/master/images/3.jpg)

> SPP net对上述公式进行了简化，令每一层的padding都为：<a href="https://www.codecogs.com/eqnedit.php?latex=padding&space;=&space;[\frac{k_{i}}{2}]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?padding&space;=&space;[\frac{k_{i}}{2}]" title="padding = [\frac{k_{i}}{2}]" /></a>，则<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}&space;&plus;&space;\frac{k_{i}-1}{2}&space;-&space;[\frac{k_{i}}{2}]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}&space;&plus;&space;\frac{k_{i}-1}{2}&space;-&space;[\frac{k_{i}}{2}]" title="p_{i} = s_{i}\cdot p_{i+1} + \frac{k_{i}-1}{2} - [\frac{k_{i}}{2}]" /></a>

> 当<a href="https://www.codecogs.com/eqnedit.php?latex=k_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k_{i}" title="k_{i}" /></a>为奇数时，<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{k_{i}-1}{2}&space;-&space;[\frac{k_{i}}{2}]&space;=&space;0&space;\Rightarrow&space;p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\frac{k_{i}-1}{2}&space;-&space;[\frac{k_{i}}{2}]&space;=&space;0&space;\Rightarrow&space;p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}" title="\frac{k_{i}-1}{2} - [\frac{k_{i}}{2}] = 0 \Rightarrow p_{i} = s_{i}\cdot p_{i+1}" /></a>

> 当<a href="https://www.codecogs.com/eqnedit.php?latex=k_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k_{i}" title="k_{i}" /></a>为偶数时，<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{k_{i}-1}{2}&space;-&space;[\frac{k_{i}}{2}]&space;=&space;-0.5&space;\Rightarrow&space;p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}&space;-&space;0.5" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\frac{k_{i}-1}{2}&space;-&space;[\frac{k_{i}}{2}]&space;=&space;-0.5&space;\Rightarrow&space;p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}&space;-&space;0.5" title="\frac{k_{i}-1}{2} - [\frac{k_{i}}{2}] = -0.5 \Rightarrow p_{i} = s_{i}\cdot p_{i+1} - 0.5" /></a>

> 由于<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p_{i}" title="p_{i}" /></a>为坐标值，为整数，所以可以得到：<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p_{i}&space;=&space;s_{i}\cdot&space;p_{i&plus;1}" title="p_{i} = s_{i}\cdot p_{i+1}" /></a>，感受野中心点的坐标<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p_{i}" title="p_{i}" /></a>只跟前一层<a href="https://www.codecogs.com/eqnedit.php?latex=p_{i&plus;1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p_{i&plus;1}" title="p_{i+1}" /></a>有关，对该公式进行级联得到：<a href="https://www.codecogs.com/eqnedit.php?latex=p_{0}&space;=&space;S\cdot&space;p_{i&plus;1},\&space;S&space;=&space;\prod_{0}^{i}s_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p_{0}&space;=&space;S\cdot&space;p_{i&plus;1},\&space;S&space;=&space;\prod_{0}^{i}s_{i}" title="p_{0} = S\cdot p_{i+1},\ S = \prod_{0}^{i}s_{i}" /></a>

### 3. 候选区域到feature map的特征映射

> SPP net是将原始的ROI的左上角和右下角映射到feature map上的两个对应点， 有了feature map上的两对角点就确定了对应的 feature map 区域，如图所示：

![image](https://github.com/ShaoQiBNU/SPPnet/blob/master/images/4.jpg)

> 左上角的点<a href="https://www.codecogs.com/eqnedit.php?latex=(x,y)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(x,y)" title="(x,y)" /></a>映射到feature map上的<a href="https://www.codecogs.com/eqnedit.php?latex=({x}',{y}')" target="_blank"><img src="https://latex.codecogs.com/svg.latex?({x}',{y}')" title="({x}',{y}')" /></a>，根据2中的公式可以得到转换关系：<a href="https://www.codecogs.com/eqnedit.php?latex=(x,y)&space;=&space;(S{x}',S{y}')" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(x,y)&space;=&space;(S{x}',S{y}')" title="(x,y) = (S{x}',S{y}')" /></a>。根据<a href="https://www.codecogs.com/eqnedit.php?latex=(x,y)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(x,y)" title="(x,y)" /></a>求算<a href="https://www.codecogs.com/eqnedit.php?latex=({x}',{y}')" target="_blank"><img src="https://latex.codecogs.com/svg.latex?({x}',{y}')" title="({x}',{y}')" /></a>，则计算公式如下：<a href="https://www.codecogs.com/eqnedit.php?latex={x}'&space;=&space;[\frac{x}{S}]&plus;1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{x}'&space;=&space;[\frac{x}{S}]&plus;1" title="{x}' = [\frac{x}{S}]+1" /></a>，各个角点的计算公式最终确定如下：
>
> 左上角：<a href="https://www.codecogs.com/eqnedit.php?latex={x}'&space;=&space;[\frac{x}{S}]&plus;1,{y}'&space;=&space;[\frac{y}{S}]&plus;1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{x}'&space;=&space;[\frac{x}{S}]&plus;1,{y}'&space;=&space;[\frac{y}{S}]&plus;1" title="{x}' = [\frac{x}{S}]+1,{y}' = [\frac{y}{S}]+1" /></a>
>
> 右下角：<a href="https://www.codecogs.com/eqnedit.php?latex={x}'&space;=&space;[\frac{x}{S}]-1,{y}'&space;=&space;[\frac{y}{S}]-1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?{x}'&space;=&space;[\frac{x}{S}]-1,{y}'&space;=&space;[\frac{y}{S}]-1" title="{x}' = [\frac{x}{S}]-1,{y}' = [\frac{y}{S}]-1" /></a>


