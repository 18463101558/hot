# coding=utf-8
import math
import numpy as np
import tensorflow as tf
from functools import reduce
class ResNet:
    def __init__(self, ResNet_npy_path=None, trainable=True, open_tensorboard=False):
        """
         初始化函数
         ResNet_npy_path：如果路径不是none，则加载模型。 否则，随机初始化所有参数。
         open_tensorboard：是否打开Tensorboard。
        """
        self.var_dict = {}
        self.trainable = trainable
        self.open_tensorboard = open_tensorboard
        self.is_training = True
        self.data_dict=None
    def set_is_training(self, isTrain):
        """
        Set is training bool.
        """
        self.is_training = isTrain

    def build(self, inputs, label_num):
        self.conv1 = self.conv_layer(inputs, 7, 1, 64, 2, "conv1")#复用成功
        self.conv_norm_1 = self.batch_norm(self.conv1)#conv
        self.conv1_relu = tf.nn.relu(self.conv_norm_1)
        self.pool1 = self.max_pool(self.conv1_relu, 1, 2, "pool1")#relu，也就是自己重写的relu

        self.block1_1 = self.res_block_3_layers(self.pool1, [64, 64, 256], "block1_1", True)#4次复用卷积成功
        self.block1_2 = self.res_block_3_layers(self.block1_1, [64, 64, 256], "block1_2")#3次复用
        self.block1_3 = self.res_block_3_layers(self.block1_2, [64, 64, 256], "block1_3")#3次复用

        self.block2_1 = self.res_block_3_layers(self.block1_3, [128, 128, 512], "block2_1", True, 2)#4次复用卷积成功
        self.block2_2 = self.res_block_3_layers(self.block2_1, [128, 128, 512], "block2_2")#3次复用成功
        self.block2_3 = self.res_block_3_layers(self.block2_2, [128, 128, 512], "block2_3")#3次复用成功
        self.block2_4 = self.res_block_3_layers(self.block2_3, [128, 128, 512], "block2_4")#3次

        self.block3_1 = self.res_block_3_layers(self.block2_4, [256, 256, 1024], "block3_1", True, 2)#4次
        self.block3_2 = self.res_block_3_layers(self.block3_1, [256, 256, 1024], "block3_2")#3次
        self.block3_3 = self.res_block_3_layers(self.block3_2, [256, 256, 1024], "block3_3")#3次
        self.block3_4 = self.res_block_3_layers(self.block3_3, [256, 256, 1024], "block3_4")#3次
        self.block3_5 = self.res_block_3_layers(self.block3_4, [256, 256, 1024], "block3_5")#3次
        self.block3_6 = self.res_block_3_layers(self.block3_5, [256, 256, 1024], "block3_6")#3次

        self.block4_1 = self.res_block_3_layers(self.block3_6, [512, 512, 2048], "block4_1", True, 2)#4次
        self.block4_2 = self.res_block_3_layers(self.block4_1, [512, 512, 2048], "block4_2")#3次
        self.block4_3 = self.res_block_3_layers(self.block4_2, [512, 512, 2048], "block4_3")#3次重用
        self.pool2 = self.avg_pool(self.block4_3, 4,2, "pool2")#其实这里就是一个简单的平均池化
        self.fc1 = self.fc_layer(self.pool2,  1, "fc1")#全连接层来也，1次重用，就是权重和偏置都重用了
        self.prob = tf.nn.sigmoid(self.fc1, name="prob")

        return self.prob


    def res_block_3_layers(self, bottom, channel_list, name, change_dimension = False, block_stride = 1):
        """
        bottom: input values (X)
        channel_list : number of channel in 3 layers
        name: block name
        """
        if (change_dimension):#如果有转换维度的命令，第一个是输入值，第二个是核大小，也就是直接卷积，这一个是用于远程连接的，用于缩小featruemap

            short_cut_conv = self.conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[2], block_stride, name + "_ShortcutConv")
            block_conv_input = self.batch_norm(short_cut_conv)#进行归一化
        else:
            block_conv_input = bottom

        block_conv_1 = self.conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[0], block_stride, name + "_lovalConv1")
        block_norm_1 = self.batch_norm(block_conv_1)
        block_relu_1 = tf.nn.relu(block_norm_1)

        block_conv_2 = self.conv_layer(block_relu_1, 3, channel_list[0], channel_list[1], 1, name + "_lovalConv2")
        block_norm_2 = self.batch_norm(block_conv_2)
        block_relu_2 = tf.nn.relu(block_norm_2)

        block_conv_3 = self.conv_layer(block_relu_2, 1, channel_list[1], channel_list[2], 1, name + "_lovalConv3")
        block_norm_3 = self.batch_norm(block_conv_3)
        block_res = tf.add(block_conv_input, block_norm_3)#进行tensorflow的加和连接
        relu = tf.nn.relu(block_res)

        return relu

    def batch_norm(self, inputsTensor):
        """
        Batchnorm
        """
        _BATCH_NORM_DECAY = 0.99#动量，也就是那个比值，应该就是α和γ吧
        _BATCH_NORM_EPSILON = 1e-12#防止除0错误的东东
        return tf.layers.batch_normalization(inputs=inputsTensor, axis = 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.is_training )

    def avg_pool(self, bottom, kernal_size = 2, stride = 2, name = "avg"):
        """
        bottom: input values (X)
        kernal_size : n * n kernal
        stride : stride
        name : block_layer name
        """
        print(name + ":")
        print(bottom.get_shape().as_list())
        return tf.nn.avg_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    def max_pool(self, bottom, kernal_size = 2, stride = 2, name = "max"):
        """
        bottom: input values (X)
        kernal_size : n * n kernal
        stride : stride
        name : block_layer name
        """
        print(name ,":",bottom.get_shape().as_list())
        return tf.nn.max_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, kernal_size, in_channels, out_channels, stride, name):
        """
         底部：输入值（X）
         kernal_size：n * n kernal
         in_channels：输入过滤器的数量
         out_channels：输出过滤器的数量
         迈步：迈步
         名称：block_layer名称
        """
        print(name , ":",bottom.get_shape().as_list())
        with tf.variable_scope(name):#定义变量作用域，以name作为标志
            filt, conv_biases = self.get_conv_var(kernal_size, in_channels, out_channels, name)#获取权重和偏置，如果没有权重和偏置，那么生成新的
            conv = tf.nn.conv2d(bottom, filt, [1,stride,stride,1], padding='SAME')
            #源API为：tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
            #input为[batch, in_height, in_width, in_channels]
            #filter为[filter_height, filter_width, in_channels, out_channels]
            #strides = [1, stride, stride, 1]
            bias = tf.nn.bias_add(conv, conv_biases)#加上偏置

            tf.summary.histogram('weight', filt)
            tf.summary.histogram('bias', conv_biases)
            #tf.summary.histogram（）将输入的一个任意大小和形状的张量压缩成一个由宽度和数量组成的直方图数据结构
            #也就是记录张量训练值的分布情况，用一个直方图绘制出来
            return bias

    def fc_layer(self, bottom,  out_size, name):
        """
        bottom: input values (X)
        in_size : number of input feature size
        out_size : number of output feature size
        """
        #print(name + ":")
        tensorshape=bottom.get_shape().as_list()
        in_size=tensorshape[1]*tensorshape[2]*tensorshape[3]
        #print("测试insize的大小：",in_size)
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            tf.summary.histogram('weight', weights)
            tf.summary.histogram('bias', biases)

        return fc
    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        """
        filter_size : 3 * 3
        in_channels : 输入通道
        out_channels :输出通道
        name : block_layer的名称
        """
        #这里的truncated_normal指的是初始化的方法为高斯随机初始化，随机初始化滤波器大小
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, stddev = 1 / math.sqrt(float(filter_size * filter_size)))
        #print("复用滤波器！")
        filters = self.get_var(initial_value, name, 0, name + "_filters")
        #这个函数的作用就是检查变量有没有用过，如果用过的话，那么可以直接从原来已有的变量里面重用它

        #随机初始化偏置大小
        #print("复用偏置！")
        initial_value = tf.truncated_normal([out_channels], 0.0, 1.0)
        biases = self.get_var(initial_value, name, 1, name + "_biases")#尝试重用变量

        return filters, biases
    def get_fc_var(self, in_size, out_size, name):
        """
        in_size : number of input feature size
        out_size : number of output feature size
        name : block_layer name
        """
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, stddev = 1 / math.sqrt(float(in_size)))
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, 1.0)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases
    def get_var(self, initial_value, name, idx, var_name):
        """
         从Loaded模型或新生成的随机变量加载变量
         initial_value：随机初始化值
         name：block_layer名称
         idx：0,1分别代表权重和偏置
         var_name：name +“_filter”/“_ bias”
        """
        value = initial_value
        var = tf.Variable(value, name=var_name)
        return var


    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

