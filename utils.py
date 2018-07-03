# coding=utf-8
from scipy import misc
import scipy.io as scio
import tensorflow as tf
import numpy as np
import time
import math
import random
import csv
import os

def random_mini_batches(totalSize, mini_batch_size = 64, random = True):
    np.random.seed(int(time.time()))        
    m = totalSize
    mini_batches = []

    if(random):#如果打乱的话
        permutation = list(np.random.permutation(m))
        #函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）；
        # 区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
        # 而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
    else:
        permutation = list(range(m))

    num_complete_minibatches = math.floor(m/mini_batch_size) # math.floor(x)返回小于参数x的最大整数,即对浮点数向下取整
    for k in range(0, num_complete_minibatches):
        mini_batches.append(permutation[k * mini_batch_size : (k + 1) * mini_batch_size])#直接通过贴上去的方式产生样本

    if m % mini_batch_size != 0:
        mini_batches.append(permutation[(k + 1) * mini_batch_size :])#把最后剩下的那部分样本贴上去，也就是除不断的那部分，保证这样每一张图片都能取到
    return mini_batches

def load_all_image(nameList, h, w, c,  create_npy = False):
    all_size = len(nameList)#标签里面的图像数量
    all_data = np.zeros((all_size, h, w, c), dtype = "uint8")#先预申请一个比较大的空间

    for i in range(all_size):
        print("当前选取图片",nameList[i])
        tmp_img = load_images("imgs/"+ str(nameList[i]))#加载这一张图片
        all_data[i,:,:,0] = tmp_img[:,:]#全贴上去
    all_data=all_data/255.0#对数据进行归一化
    print("图片加载至内存完成！")
    #np.save('label/imgdata.npy',all_data)#将图片保存到npy里面
    return all_data
def get_minibatch(indexList, labelList, h, w, c, n, allImage):
    """
     加载一个批次图像。
     indexList：（size，1）。
     nameList：（totalSize，string）。
     labelList：（totalSize，int）
     h，w，c：高度，宽度，通道
     n：标签数量
    """
    m_size = len(indexList)
    batch_X = np.ndarray([m_size, h, w, c])
    batch_Y = np.zeros((m_size, n))
    for i in range(m_size):
        batch_X[i,:,:,:] = allImage[indexList[i],:,:,:]
        batch_Y[i, :] = labelList[indexList[i]]
    return batch_X, batch_Y
def read_file_list(filename):
    trainNameList=[]
    trainLabelList=[]
    csvFile=open(filename,encoding='utf-8')
    csv_reader = csv.reader(csvFile)
    for row in csv_reader:
        trainNameList.append(str(row[0]))#图片名称列表
        trainLabelList.append(int(row[1]))#图片所对应的标签
    csvFile.close()
    return trainNameList,trainLabelList

def load_images(path):
    img = misc.imread(path).astype(float)
    return img
def removefile(filename):
    if os.path.isfile( filename):
        try:
            os.remove( filename)
        except  Exception as e:
            print(e)
def save_file_list(filename,ImageNamelist,LabelList):
    csvFile = open(filename, "w", newline='',encoding='utf-8')
    writer = csv.writer(csvFile)
    for i in range (len(ImageNamelist)):
        writer.writerow([ImageNamelist[i], LabelList[i]])
    csvFile.close()
def  preparedata():
    removefile("trainlist.csv")
    removefile("validationlist.csv")
    fileNameList, fileLabelList = read_file_list('data.csv')
    print("在解码之前-------------------------------------------")
    for name in fileNameList:
        print("图片名称",name)
    trainList=[]
    trainLabelList=[]
    valList=[]
    valLabelList=[]
    for i in range (len(fileLabelList)):
        rand= random.randint(1, 5)
        if rand==5:
            valList.append(str(fileNameList[i]))
            valLabelList.append(str(fileLabelList[i]))
        else:
            trainList.append(str(fileNameList[i]))
            trainLabelList.append(str(fileLabelList[i]))
    save_file_list("trainlist.csv",trainList,trainLabelList)
    save_file_list("validationlist.csv",valList,valLabelList)
def compute_standard(minibatch_Y,recordprob,recordcost,total_cost,total_count,total_TP,total_FP,total_FN,total_TN):
    total_cost += recordcost
    total_count += len(recordprob)
    print("prob:",recordprob,"cost",recordcost)
    recordprob[recordprob >= 0.5] = 1
    recordprob[recordprob < 0.5] = 0
    TP = np.sum(np.logical_and(np.equal(minibatch_Y, 1), np.equal(recordprob, 1)))  # 正例并且识别为正类
    FP = np.sum(np.logical_and(np.equal(minibatch_Y, 0), np.equal(recordprob, 1)))  # 负例识别成正例
    FN = np.sum(np.logical_and(np.equal(minibatch_Y, 1), np.equal(recordprob, 0)))  # 正例识别成负例
    TN = np.sum(np.logical_and(np.equal(minibatch_Y, 0), np.equal(recordprob, 0)))  # 负例识别成负例
    print("TP:", TP ,",FP", FP,",FN", FN,",TN", TN,)
    total_TP += TP
    total_FP += FP
    total_FN += FN
    total_TN += TN
    return total_cost,total_count,total_TP,total_FP,total_FN,total_TN