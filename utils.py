"""
Some useful functions are defined in this file
Author: Kaihua Tang
"""
from scipy import misc
import scipy.io as scio
import tensorflow as tf
import numpy as np
import time
import math
import random


def random_mini_batches(totalSize, mini_batch_size = 64, random = True):
    """
    176/5000
     totalSize：训练图像的总数
     mini_batch_size：迷你批量大小
     返回一组包含从1到totalSize的索引的数组，每个数组为mini_batch_size
    """
    np.random.seed(int(time.time()))        
    m = totalSize                   # number of training examples
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
        mini_batches.append(permutation[k * mini_batch_size : (k + 1) * mini_batch_size])
    #直接通过贴上去的方式产生样本
    if m % mini_batch_size != 0:
        mini_batches.append(permutation[(k + 1) * mini_batch_size :])#把最后剩下的那部分样本贴上去，也就是除不断的那部分，保证这样每一张图片都能取到
    return mini_batches

def load_all_image(nameList, h, w, c, parentPath, create_npy = False):
    """
    Load all image data in advance
    nameList: name of image we need to load
    """
    all_size = len(nameList)#标签里面的图像数量
    all_data = np.zeros((all_size, h, w, c), dtype = "uint8")#先预申请一个比较大的空间
    for i in range(all_size):
        tmp_img = load_images(parentPath + nameList[i])#加载这一张图片
        all_data[i,:,:,0] = tmp_img[:,:]#全贴上去
    print("开始保存图片")
    np.save('label/imgdata.npy',all_data)#将图片保存到npy里面
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
    print("batch_x的大小：",len(batch_X))
    print("batch_y的大小：",len(batch_Y))
    #print(paths)
    for i in range(m_size):
        batch_X[i,:,:,:] = allImage[indexList[i],:,:,:]
        batch_Y[i, :] = labelList[indexList[i]]
    return batch_X, batch_Y


def load_images(path):
    """
    Load multiple images.
    :param paths: The image paths.
    """
    img = misc.imread(path).astype(float)
    return img
