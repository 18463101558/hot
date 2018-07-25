# coding=utf-8
import ResNet as resnet
from utils import *
from conf.config import config
from DBUtil import DBUtil
import tensorflow as tf
conf=config()
model_path =conf.model_path#预加载模型位置
img_path=conf.highlocation#获取图片保存路径
HEIGHT=conf.heightborder
WIDTH=conf.widthborder
rootpath=conf.rootpath
CHANNELS=1
def analysis(record,filename):
    csvFile = open(filename, "w", newline='', encoding='utf-8')
    writer = csv.writer(csvFile)
    for temp in record:
         writer.writerow([temp[0],temp[1]])
    csvFile.close()

def get_img_list():
    NameList=[]
    LabelList=[]
    dirs = os.listdir(img_path)
    for file in dirs:
        NameList.append(file)
        LabelList.append(0)#这个不重要，贴啥都一样
    return NameList,LabelList

def Predict():
    NameList,LabelList=get_img_list()
    ImageData=load_all_image(img_path,NameList, HEIGHT, WIDTH, CHANNELS)#加载所有图片到内存里面
    num_train_image = len(NameList)#记录了所有图片数量
    sess=tf.Session()
    images = tf.placeholder(tf.float32, shape = [None,  HEIGHT, WIDTH,CHANNELS])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    # 建立网络模型
    resnet_model = resnet.ResNet(ResNet_npy_path = model_path)
    resnet_model.build(images, 1)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())#初始化所有参数
    saver.restore(sess, model_path)
    print("begin predicting!")

    record=[]
    index=0
        minibatches = random_mini_batches(num_train_image, 1, random=False)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = get_minibatch(minibatch, LabelList, HEIGHT, WIDTH, CHANNELS, 1,ImageData)
            resnet_model.set_is_training(False)
            recordprob= sess.run(resnet_model.prob, feed_dict={images: minibatch_X})
            print("这是一个批次！")
            for  onerecord in recordprob:
                print("imgname:", NameList[index],  '----------possiblity:',onerecord[0])
                saveonerecord = [str(NameList[index]), onerecord[0]]
                index = index + 1
                record.append(saveonerecord)
        print("predicting over！")
        analysis(record, 'predict的验证结果.csv')
        return record
def Save_To_Database(record):
    for onerecord in record:
         picturelocation=rootpath+img_path+"/"+onerecord[0]
         spm=onerecord[0][:-20]
         likelihood=onerecord[1]
         conn = DBUtil()
         tablename = conf.tablename
         sql ="REPLACE INTO " + tablename +" SET  spm = %s, Normal_possibility = %s, location = %s;"#注意这行语句只能在spm为主码时使用
         args =(spm,str(likelihood),picturelocation)
         conn.executesearch(sql, args)
    print("save to database over！")
if __name__ == '__main__':
    records=Predict()
    Save_To_Database(records)

