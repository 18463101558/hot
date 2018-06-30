
import math
import time
import tensorflow as tf
import ResNet as resnet
import numpy as np
import scipy.io as scio
from scipy import misc
from utils import *
import csv
# image size
WIDTH = 360
HEIGHT = 120
CHANNELS = 1
#"Mini batch size"
MINI_BATCH_SIZE = 32
#"Path of Label.npy"
label_path = "./label/label_1200.npy"
#"Path of image file names"
image_name_path = "./label/name_1200.npy"
# image path
parentPath = "imgs/"
# data Path: n * 224 * 224 * 3 numpy matrix
data_path = None



def Train():
    """
   网络的超参数
     model_path：预训练模型的路径，如果没有这样的模型，则设置None。
     LABELSNUM：输出标签的数量
     learning_rate_orig：原始学习率
     NUM_EPOCHS：时代数量
     save_frequency：保存模型的频率（epoches的数量）
    """
    model_path ="./model/03.npy"

    learning_rate_orig = 1e-05
    NUM_EPOCHS = 1000
    save_frequency = 2
    """
     分类层
     final_layer_type：softmax或sigmoid
     is_sparse：当最后一层是softmax时，它是稀疏的
    """
    """
     Tensorboard设置
     tensorboard_on：打开Tensorboard或不打开
     TensorBoard_refresh：刷新率（批次数）
     monitoring_rate：打印输出速率
    """
    tensorboard_on = False
    TensorBoard_refresh = 50
    monitoring_rate = 50#每多少个epoch执行一次监控
    trainNameList=[]
    trainLabelList=[]
    csv_reader = csv.reader(open('data.csv', encoding='utf-8'))
    for row in csv_reader:
        trainNameList.append(row[0])
        trainLabelList.append(int(row[1]))

    #allImageData读取的是所有图片的数据集合
    allImageData = load_all_image(trainNameList, HEIGHT, WIDTH, CHANNELS, parentPath)#没有npy图像数据，那么新建一个npy文件以保存图像数据
    #记录了所有图片数量
    num_train_image = len(trainLabelList)
    #到这里图片准备完毕

    with tf.Session() as sess:#我觉得tf中的session是指tf定义的操作吧？
        images = tf.placeholder(tf.float32, shape = [None,  HEIGHT, WIDTH,CHANNELS])#
        labels = tf.placeholder(tf.float32, shape=[None, 1])

        # 建立网络模型
        resnet_model = resnet.ResNet(ResNet_npy_path = model_path)#进行一些初始化的工作
        resnet_model.build(images, 1)

        num_minibatches = int(num_train_image / MINI_BATCH_SIZE)#计算批次,用每一批的大小去除以所有图片的数量

        # cost function
        #定义损失函数，这里已经用上label标签了
        with tf.name_scope("cost"):
            print("Using weighted sigmoid loss")#
            print(labels.shape)
            print(resnet_model.prob.shape)
            loss =labels  * tf.log(tf.clip_by_value(resnet_model.prob, 1e-10, 1.0))
            cost = tf.reduce_sum(loss)

        #定义优化函数和优化方法
        with tf.name_scope("train"):
            global_steps = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, num_minibatches * 40, 0.1, staircase = True)
            #如果staircase=True，那么每num_minibatches * 40更新一次decay_rate，global_steps代表已经迭代的次数
            #decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)。decrate为0.1
            train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

        sess.run(tf.global_variables_initializer())#初始化所有参数
        print(resnet_model.get_var_count())#统计参数数量

        # if(tensorboard_on):#设置写入tensorboard
        #     merged_summary = tf.summary.merge_all()
        #     writer = tf.summary.FileWriter("./TensorBoard/Result")
        #     writer.add_graph(sess.graph)
        #     summary_times = 0

        for epoch in range(NUM_EPOCHS):#全体epoch数量
            print("Start Epoch %i" % (epoch + 1))#开始计算epoch

            minibatch_cost = 0.0
            # count the number of batch
            batch_index = 0  
            # get index for all mini batches
            minibatches = random_mini_batches(num_train_image, MINI_BATCH_SIZE, random = True) #这里只是产生了一个随机的索引,然后把随机索引当做一个个块贴进去

            for minibatch in minibatches:
                # 第一个参数是索引列表，第二个是label的路径，返回值即为训练数据
                (minibatch_X, minibatch_Y) = get_minibatch(minibatch, trainLabelList, HEIGHT, WIDTH, CHANNELS, 1, allImageData)

                # 修改global_steps从而完成对学习率的设置
                sess.run(global_steps.assign(epoch * num_minibatches + batch_index))

                # record examples to monitoring the training process
                if((batch_index % monitoring_rate == 0)):
                    resnet_model.set_is_training(False)
                    fc1, prob = sess.run([resnet_model.fc1, resnet_model.prob], feed_dict={images: minibatch_X})
                    countMax = np.sum(np.argmax(prob,1) == minibatch_Y)#执行一次打印当前迭代次数和是第几批
                    print("Epoch %i Batch %i Before Optimization Count %i" %(epoch + 1,batch_index, countMax))
 
                # 设置resnet这个计算图可以训练
                resnet_model.set_is_training(True)
                temp_cost, _ = sess.run([cost, train], feed_dict={images: minibatch_X, labels: minibatch_Y})#输出labels，这里可训练
                minibatch_cost += np.sum(temp_cost)

                # 设置tensorboard属性
                # if(tensorboard_on) and (batch_index % TensorBoard_refresh == 0):
                #     s = sess.run(merged_summary, feed_dict={images: minibatch_X, labels: minibatch_Y})
                #     writer.add_summary(s, summary_times)
                #     summary_times = summary_times + 1
                #     # record cost in tensorflow
                #     tf.summary.scalar('cost', temp_cost)
                
                # # 记录当前的准确率
                # if((batch_index % monitoring_rate == 0)):
                #     resnet_model.set_is_training(False)
                #     fc1, prob = sess.run([resnet_model.fc1, resnet_model.prob], feed_dict={images: minibatch_X})
                #     countMax = np.sum(np.argmax(prob,1) == minibatch_Y)
                #     print("Epoch %i Batch %i After Optimization Count %i" %(epoch + 1,batch_index, countMax))
                #     print("Epoch %i Batch %i Batch Cost %f Learning_rate %f" %(epoch + 1,batch_index, np.sum(temp_cost), sess.run(learning_rate) * 1e10))

                batch_index += 1


            # print total cost of this epoch
            print("End Epoch %i" % (epoch + 1))
            print("Total cost of Epoch %f" % minibatch_cost)

            # save model
            if((epoch + 1) % save_frequency == 0):
                resnet_model.save_npy(sess, './model/temp-model%i.npy' % (epoch + 1))

if __name__ == '__main__':
    Train()
