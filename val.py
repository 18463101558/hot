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
MINI_BATCH_SIZE =1
model_path =conf.model_path

def analysis(record,filename):
    csvFile = open(filename, "w", newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    for temp in record:
         writer.writerow([temp[0],temp[1],temp[2][0]])
    csvFile.close()

def Train():
    trainNameList, trainLabelList=read_file_list('trainlist.csv')
    valNameList, valLabelList = read_file_list('validationlist.csv')
    testNameList, testLabelList = read_file_list('testlist.csv')

    TrainImageData = load_all_image(conf.trainlocation,trainNameList, HEIGHT, WIDTH, CHANNELS)
    ValImageData=load_all_image(conf.trainlocation,valNameList, HEIGHT, WIDTH, CHANNELS)
    TestImageData=load_all_image(conf.trainlocation,testNameList, HEIGHT, WIDTH, CHANNELS)

    num_train_image = len(trainLabelList)#记录了所有图片数量
    num_val_image = len(valLabelList)
    num_test_image=len(testLabelList)

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, shape = [None,  HEIGHT, WIDTH,CHANNELS])
        labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 建立网络模型
        resnet_model = resnet.ResNet(ResNet_npy_path = model_path)#进行一些初始化的工作
        resnet_model.build(images, 1)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())#初始化所有参数
        saver.restore(sess, model_path)
        print("开始验证！")

        trainrecord=[]
        valrecord=[]
        testrecord = []

        index=0
        minibatches = random_mini_batches(num_train_image, MINI_BATCH_SIZE, random=False)  # 这里是产生了很多大小为batch_size的块
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = get_minibatch(minibatch, trainLabelList, HEIGHT, WIDTH, CHANNELS, 1,TrainImageData)
            resnet_model.set_is_training(True)
            recordprob= sess.run(resnet_model.prob,
                                                 feed_dict={images: minibatch_X})  # 输出labels，这里可训练
            for  onerecord in recordprob:
                    print("图像名称为:", trainNameList[index], "-----------标签为:", trainLabelList[index], '----------预测值为:',onerecord)
                    saveonerecord = [str(trainNameList[index]), str(trainLabelList[index]), onerecord]
                    index = index + 1
                    trainrecord.append(saveonerecord)
        print("训练集预测完成！")

        index=0
        minibatches = random_mini_batches(num_val_image, MINI_BATCH_SIZE, random=False)  # 这里是产生了很多大小为batch_size的块
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = get_minibatch(minibatch, valLabelList, HEIGHT, WIDTH, CHANNELS, 1,ValImageData)
            resnet_model.set_is_training(True)
            recordprob= sess.run(resnet_model.prob,
                                                 feed_dict={images: minibatch_X})  # 输出labels，这里可训练
            for  onerecord in recordprob:
                    print("图像名称为:", valNameList[index], "-----------标签为:", valLabelList[index], '----------预测值为:',onerecord)
                    saveonerecord = [str(valNameList[index]), str(valLabelList[index]),onerecord]
                    index = index + 1
                    valrecord.append(saveonerecord)
        print("验证集预测完成！")

        index=0
        minibatches = random_mini_batches(num_test_image, MINI_BATCH_SIZE, random=False)  # 这里是产生了很多大小为batch_size的块
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = get_minibatch(minibatch, testLabelList, HEIGHT, WIDTH, CHANNELS, 1,TestImageData)
            resnet_model.set_is_training(True)
            recordprob= sess.run(resnet_model.prob,
                                                 feed_dict={images: minibatch_X})  # 输出labels，这里可训练
            for  onerecord in recordprob:
                    print("图像名称为:", testNameList[index], "-----------标签为:", testLabelList[index], '----------预测值为:',onerecord)
                    saveonerecord = [str(testNameList[index]), str(testLabelList[index]),onerecord]
                    index = index + 1
                    testrecord.append(saveonerecord)
        print("测试集预测完成！")

        analysis(trainrecord,'train的验证结果.csv')
        analysis(valrecord, 'val的验证结果.csv')
        analysis(testrecord, 'test的验证结果.csv')
if __name__ == '__main__':
    Train()

