
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

def Train():


    trainNameList, trainLabelList=read_file_list('trainlist.csv')
    valNameList, valLabelList = read_file_list('validationlist.csv')

    TrainImageData = load_all_image(trainNameList, HEIGHT, WIDTH, CHANNELS)
    ValImageData=load_all_image(valNameList, HEIGHT, WIDTH, CHANNELS)
    num_train_image = len(trainLabelList)#记录了所有图片数量
    #到这里图片准备完毕
    num_val_image=len(valLabelList )
    num_minibatches = int(num_train_image / MINI_BATCH_SIZE)  # 计算每一个epoch批次,用每一批的大小去除以所有图片的数量

    with tf.Session() as sess:#我觉得tf中的session是指tf定义的操作吧？
        images = tf.placeholder(tf.float32, shape = [None,  HEIGHT, WIDTH,CHANNELS])#
        labels = tf.placeholder(tf.float32, shape=[None, 1])

        # 建立网络模型
        resnet_model = resnet.ResNet(ResNet_npy_path = model_path)#进行一些初始化的工作
        resnet_model.build(images, 1)

        #定义损失函数，这里已经用上label标签了，使用了对数损失
        with tf.name_scope("cost"):
            loss = -labels* tf.log(tf.clip_by_value(resnet_model.prob,1e-10,1.0)) - (1-labels) * tf.log(tf.clip_by_value(1-resnet_model.prob,1e-10,1.0))#使用二分类对数损失
            cost = tf.reduce_mean(loss)

        #定义优化函数和优化方法
        with tf.name_scope("train"):
            global_steps = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, num_minibatches * 40, 0.1, staircase = True)
            #如果staircase=True，那么每num_minibatches * 40更新一次decay_rate，global_steps代表已经迭代的次数
            #decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)。decrate为0.1
            train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

        sess.run(tf.global_variables_initializer())#初始化所有参数
        print("网络结构定义完毕，总共参数数量为：",resnet_model.get_var_count())#统计参数数量

        merged_summary =tf.summary.merge_all()
        writer = tf.summary.FileWriter("./TensorBoard/Result")
        writer.add_graph(sess.graph)#生成数据流图

        for epoch in range(NUM_EPOCHS):#全体epoch数量
            total_cost = 0.0
            batch_index = 0
            total_count=0
            total_TP = 0
            total_FP=0
            total_FN=0
            total_TN=0
            print("Start Epoch %i" % (epoch + 1))  # 开始计算epoch
            minibatches = random_mini_batches(num_train_image, MINI_BATCH_SIZE, random = True)#这里是产生了很多大小为batch_size的块

            for minibatch in minibatches:
                # 第一个参数是索引列表，第二个是label的路径，返回值即为训练数据
                (minibatch_X, minibatch_Y) = get_minibatch(minibatch, trainLabelList, HEIGHT, WIDTH, CHANNELS, 1,TrainImageData)
                # 修改global_steps从而完成对学习率的设置，这里相当于在每一个epoch之前设置学习率
                sess.run(global_steps.assign(epoch * num_minibatches + batch_index))
                # 设置resnet这个计算图可以训练,因为这里train才设置了优化目标，所以这里必须将train包含进去
                resnet_model.set_is_training(True)
                recordcost, recordprob,_= sess.run([cost, resnet_model.prob,train], feed_dict={images: minibatch_X, labels: minibatch_Y})#输出labels，这里可训练
                total_cost, total_count, total_TP, total_FP, total_FN, total_TN=compute_standard(minibatch_Y, recordprob, recordcost, total_cost, total_count, total_TP, total_FP,
                                 total_FN, total_TN)

                if ((batch_index % monitoring_rate == 0)):
                    acc = (total_TP+total_TN)/(total_TP+total_FP+total_FN+total_TN)  # 计算损失函数
                    F1_score=(2*total_TP)/(2*total_TP+total_FP+total_FN)
                    average_cost=total_cost/total_count
                    print("average_cost=%.3f,acc=%.2f, F1_score=%.2f" %(average_cost,acc,F1_score))
                    if(batch_index==(len(minibatches)-1)):#到达吗，末尾时就保存到tensorboard
                        s = sess.run(merged_summary, feed_dict={images: minibatch_X, labels: minibatch_Y})
                        writer.add_summary(s, epoch)
                        tf.summary.scalar('traincost', average_cost)
                        tf.summary.scalar('trainacc', acc)
                        tf.summary.scalar('trainF1',F1_score)
                batch_index += 1

            print("End Epoch %i" % (epoch + 1))
            val_minibatches = random_mini_batches(num_val_image, 1, random=True)#准备在验证集上面跑结果
            val_total_cost = 0.0
            val_total_count=0
            val_total_TP = 0
            val_total_FP=0
            val_total_FN=0
            val_total_TN=0
            for val_minibatch in val_minibatches:
                (val_minibatch_X, val_minibatch_Y) = get_minibatch(val_minibatch, valLabelList, HEIGHT, WIDTH,
                                                                   CHANNELS, 1,ValImageData)
                resnet_model.set_is_training(False)
                val_recordcost, val_recordprob, _ = sess.run([cost, resnet_model.prob, train],
                                                 feed_dict={images: val_minibatch_X, labels:val_minibatch_Y})  # 输出labels，这里可训练
                val_total_cost, val_total_count, val_total_TP, val_total_FP, val_total_FN, val_total_TN = compute_standard(val_minibatch_Y, val_recordprob,
                                                                                                                       val_recordcost, val_total_cost,
                                                                                                                       val_total_count, val_total_TP,
                                                                                                                       val_total_FP,val_total_FN, val_total_TN)
            val_acc = (val_total_TP + val_total_TN) / (val_total_TP + val_total_FP + val_total_FN + val_total_TN)  # 计算损失函数
            val_F1_score = (2 * val_total_TP) / (2 * val_total_TP + val_total_FP + val_total_FN)
            val_average_cost = val_total_cost / val_total_count
            print("average_cost=%.3f,acc=%.2f, F1_score=%.2f" % (average_cost, acc, F1_score))

            s = sess.run(merged_summary, feed_dict={images: minibatch_X, labels: minibatch_Y})
            writer.add_summary(s, epoch)
            tf.summary.scalar('val_traincost', val_average_cost)
            tf.summary.scalar('val_trainacc', val_acc)
            tf.summary.scalar('wal_trainF1', val_F1_score)
            # 保存模型
            if((epoch + 1) % save_frequency == 0):
                resnet_model.save_npy(sess, './model/temp-model%i.npy' % (epoch + 1))

if __name__ == '__main__':
    preparedata()
    Train()
