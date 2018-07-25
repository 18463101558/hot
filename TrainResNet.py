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
revert_flag=conf.revert_flag

def Train():
    trainNameList, trainLabelList=read_file_list('trainlist.csv')
    valNameList, valLabelList = read_file_list('validationlist.csv')
    TrainImageData = load_all_image(conf.trainlocation,trainNameList, HEIGHT, WIDTH, CHANNELS)
    ValImageData=load_all_image(conf.trainlocation,valNameList, HEIGHT, WIDTH, CHANNELS)
    # 到这里图片准备完毕

    num_train_image = len(trainLabelList)#记录了所有训练集图片数量
    num_val_image=len(valLabelList )#记录了所有验证集图片数量
    num_minibatches = int(num_train_image / MINI_BATCH_SIZE)  # 计算每一个epoch批次,用每一批的大小去除以所有图片的数量
    print("now  path is:",os.getcwd())


    #开始进入tensorflow计算图
    images = tf.placeholder(tf.float32, shape = [None,  HEIGHT, WIDTH,CHANNELS])
    labels = tf.placeholder(tf.float32, shape=[None, 1])

    # 建立网络模型
    resnet_model = resnet.ResNet(ResNet_npy_path = model_path)#进行一些初始化的工作
    resnet_model.build(images, 1)

    #定义损失函数，这里已经用上label标签了，使用了对数损失
    with tf.name_scope("cost"):
          loss = -labels* tf.log(tf.clip_by_value(resnet_model.prob,1e-10,1.0)) - (1-labels) * tf.log(tf.clip_by_value(1-resnet_model.prob,1e-10,1.0))#使用二分类对数损失
          cost = tf.reduce_sum(loss)

    #定义优化函数和优化方法
    with tf.name_scope("train"):
            global_steps = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, num_minibatches * 40, 0.1, staircase = True)
            #如果staircase=True，那么每num_minibatches * 40更新一次decay_rate，global_steps代表已经迭代的次数
            #decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)。decrate为0.1
            train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

    #定义tensorboard
    with tf.name_scope("record"):
            flag=tf.placeholder(tf.int32)
            record_acc= tf.placeholder("float")
            record_F1score = tf.placeholder("float")
            record_cost= tf.placeholder("float")
            if(flag==0):
                tf.summary.scalar('trainF1',record_F1score)
                tf.summary.scalar('trainacc',record_acc)
                tf.summary.scalar('traincost',record_cost)
            else:
                tf.summary.scalar('val_trainF1',record_F1score)
                tf.summary.scalar('val_trainacc',record_acc)
                tf.summary.scalar('val_traincost',record_cost)
    merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver()  # 默认收集所有变量

    with tf.name_scope("set_learningrate"):#设置globalsteps用
            scheduleepoch = tf.placeholder("int32")
            schedulenum_minibatches = tf.placeholder("int32")
            schedulebatch_index= tf.placeholder("int32")
            renewsteps = tf.assign(global_steps, scheduleepoch * schedulenum_minibatches + schedulebatch_index)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./TensorBoard/train', sess.graph)  # 定义写入train里面的记录数据
        val_writer = tf.summary.FileWriter('./TensorBoard/test')  # 定义写入val里面的数据

        sess.run(tf.global_variables_initializer())#初始化所有参数
        sess.graph.finalize()#锁定图，使之只能读不能写,避免后面添加节点导致出错

        if revert_flag==1:
            saver.restore(sess, model_path)
            print("revert mode!")
        for epoch in range(NUM_EPOCHS):#全体epoch数量
            total_cost = 0.0
            batch_index = total_count=total_TP =total_FP=total_FN=total_TN=0
            print("Start Epoch %i" % (epoch + 1))  # 开始计算epoch
            minibatches = random_mini_batches(num_train_image, MINI_BATCH_SIZE, random = True)#这里是产生了很多大小为batch_size的块
            for minibatch in minibatches:
                # 第一个参数是索引列表，第二个是label的路径，返回值即为训练数据
                (minibatch_X, minibatch_Y) = get_minibatch(minibatch, trainLabelList, HEIGHT, WIDTH, CHANNELS, 1,TrainImageData)
                # 修改global_steps从而完成对学习率的设置，这里相当于在每一个epoch之前设置学习率,注意这里不能直接调用assign，否则在循环里面会增加节点数量导致崩溃
                sess.run(renewsteps, feed_dict={scheduleepoch:epoch,schedulenum_minibatches:num_minibatches,schedulebatch_index:batch_index})
                # 设置resnet这个计算图可以训练,因为这里train才设置了优化目标，所以这里必须将train包含进去
                resnet_model.set_is_training(True)
                recordcost, recordprob,_= sess.run([cost, resnet_model.prob,train], feed_dict={images: minibatch_X, labels: minibatch_Y})#输出labels，这里可训练
                total_cost, total_count, total_TP, total_FP, total_FN, total_TN=compute_standard(minibatch_Y, recordprob, recordcost, total_cost, total_count, total_TP, total_FP,
                                 total_FN, total_TN)
                if ((batch_index % monitoring_rate == 0)):
                    acc = (total_TP+total_TN)/(total_TP+total_FP+total_FN+total_TN)  # 计算损失函数
                    denominator= (2 * total_TP + total_FP + total_FN)
                    if(denominator!=0):
                        F1_score = (2 * total_TP) / (2 * total_TP + total_FP + total_FN)
                    else:
                        F1_score = 1  # 也就是全部为负数的情况
                    average_cost=total_cost/total_count
                    print("total_TP:", total_TP, ",total_FP", total_FP, ",total_FN", total_FN, ",total_TN", total_TN, )
                    print("average_cost=%.3f,acc=%.2f, F1_score=%.2f" %(average_cost,acc,F1_score))
                batch_index += 1
                del  recordcost, recordprob

            print("End Epoch %i,and into validate" % (epoch + 1))
            val_minibatches = random_mini_batches(num_val_image,  MINI_BATCH_SIZE, random=False)#准备在验证集上面跑结果
            val_total_cost = 0.0
            val_total_count=val_total_TP =val_total_FP=val_total_FN=val_total_TN=0
            for val_minibatch in val_minibatches:
                (val_minibatch_X, val_minibatch_Y) = get_minibatch(val_minibatch, valLabelList, HEIGHT, WIDTH,
                                                                   CHANNELS, 1,ValImageData)
                resnet_model.set_is_training(False)
                val_recordcost, val_recordprob = sess.run([cost, resnet_model.prob],
                                                 feed_dict={images: val_minibatch_X, labels:val_minibatch_Y})  # 输出labels，这里不可训练
                val_total_cost, val_total_count, val_total_TP, val_total_FP, val_total_FN, val_total_TN = compute_standard(val_minibatch_Y, val_recordprob,
                                                                                                                       val_recordcost, val_total_cost,
                                                                                                                       val_total_count, val_total_TP,
                                                                                                                       val_total_FP,val_total_FN, val_total_TN)
                del  val_recordcost, val_recordprob
            val_acc = (val_total_TP + val_total_TN) / (val_total_TP + val_total_FP + val_total_FN + val_total_TN)  # 计算损失函数
            val_denominator=(2 * val_total_TP + val_total_FP + val_total_FN)
            if(val_denominator!=0):
                val_F1_score = (2 * val_total_TP) / (2 * val_total_TP + val_total_FP + val_total_FN)
            else:
                val_F1_score =1#也就是全部为负数的情况
            val_average_cost = val_total_cost / val_total_count
            print("val_average_cost=%.3f,val_acc=%.2f, val_F1_score=%.2f" % (val_average_cost, val_acc, val_F1_score))
            s = sess.run(merged_summary , feed_dict={flag:0,record_acc: acc,record_F1score:F1_score,record_cost:average_cost,images: minibatch_X, labels: minibatch_Y})
            train_writer.add_summary(s, epoch)
            s = sess.run(merged_summary,
                         feed_dict={flag:1, record_acc: val_acc,
                                    record_F1score: val_F1_score, record_cost: val_average_cost, images: minibatch_X,
                                    labels: minibatch_Y})
            val_writer.add_summary(s, epoch)
        saver.save(sess,model_path)

if __name__ == '__main__':
    divide_into_val_and_train()#将data里面的数据切分到验证集和训练集两个表格里面
    Train()
