# coding=utf-8
import math
import time
import tensorflow as tf
import ResNet as resnet
import numpy as np
import scipy.io as scio
from scipy import misc
from utils import *
import csv
from conf.config import config

# image size
conf=config()
WIDTH =  conf.widthborder
HEIGHT = conf.heightborder
CHANNELS = 1
MINI_BATCH_SIZE =conf.batch_size
data_path = None
learning_rate_orig = conf.learning_rate
NUM_EPOCHS =conf.epochs
model_path =conf.model_path
save_frequency =conf.save_frequency#每隔多久保存一次模型
monitoring_rate =conf.monitoring_rate#每多少次迭代计算一下打印一次损失和准确率
def analysis(record,filename):
    csvFile = open(filename, "w", newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    for temp in record:
         writer.writerow([temp[0],temp[1],temp[2][0]])
    csvFile.close()
def Train():
    trainNameList, trainLabelList=read_file_list('trainlist.csv')
    valNameList, valLabelList = read_file_list('validationlist.csv')
    TrainImageData = load_all_image(trainNameList, HEIGHT, WIDTH, CHANNELS)
    ValImageData=load_all_image(valNameList, HEIGHT, WIDTH, CHANNELS)
    num_train_image = len(trainLabelList)#记录了所有图片数量
    num_val_image = len(valLabelList)
    #到这里图片准备完毕

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, shape = [None,  HEIGHT, WIDTH,CHANNELS])
        labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 建立网络模型
        resnet_model = resnet.ResNet(ResNet_npy_path = model_path)#进行一些初始化的工作
        resnet_model.build(images, 1)

        #定义损失函数，这里已经用上label标签了，使用了对数损失
        with tf.name_scope("cost"):
            loss = -labels* tf.log(tf.clip_by_value(resnet_model.prob,1e-10,1.0)) - (1-labels) * tf.log(tf.clip_by_value(1-resnet_model.prob,1e-10,1.0))#使用二分类对数损失
            cost = tf.reduce_sum(loss)

        sess.run(tf.global_variables_initializer())#初始化所有参数
        print("net structure define,total param:",resnet_model.get_var_count())#统计参数数量
        trainrecord=[]
        valrecord=[]
        index=0
        minibatches = random_mini_batches(num_train_image, MINI_BATCH_SIZE, random=False)  # 这里是产生了很多大小为batch_size的块
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = get_minibatch(minibatch, trainLabelList, HEIGHT, WIDTH, CHANNELS, 1,TrainImageData)
            resnet_model.set_is_training(False)
            recordcost, recordprob= sess.run([cost, resnet_model.prob],
                                                 feed_dict={images: minibatch_X, labels: minibatch_Y})  # 输出labels，这里可训练
            for  onerecord in recordprob:
                    print("图像名称为:", trainNameList[index], "-----------标签为:", trainLabelList[index], '----------预测值为:',onerecord)
                    saveonerecord = [str(trainNameList[index]), str(trainLabelList[index]), onerecord]
                    index = index + 1
                    trainrecord.append(saveonerecord)
        index=0
        minibatches = random_mini_batches(num_val_image, MINI_BATCH_SIZE, random=False)  # 这里是产生了很多大小为batch_size的块
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = get_minibatch(minibatch, valLabelList, HEIGHT, WIDTH, CHANNELS, 1,ValImageData)
            resnet_model.set_is_training(False)
            recordcost, recordprob= sess.run([cost, resnet_model.prob],
                                                 feed_dict={images: minibatch_X, labels: minibatch_Y})  # 输出labels，这里可训练
            for  onerecord in recordprob:
                    print("图像名称为:", valNameList[index], "-----------标签为:", valLabelList[index], '----------预测值为:',onerecord)
                    saveonerecord = [str(valNameList[index]), str(valLabelList[index]),onerecord]
                    index = index + 1
                    valrecord.append(saveonerecord)
        analysis(trainrecord,'train的验证训练结果.csv')
        analysis(valrecord, 'val的验证训练结果.csv')
if __name__ == '__main__':
    Train()

