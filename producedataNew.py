# coding=utf-8
###############################################
#该文件将采集到的点击数据转换成图片，得到的数据被保存在hotmap.conf指定的highlocation和lowlocation路径下边
#highlocation路径下边的图片供predict预测，lowlocation路径下边下边的图片暂时不提供使用方法
###############################################
import datetime
import numpy as np
import scipy.misc
import os
import shutil
from sys import argv
from conf.config import config
from store_to_execl import  csv_operator
import re
conf=config()

def judegzero(width,height):
    if width!=0 and height!=0:
        return 1
    else:
        return 0
def judgeoutborder(width,height):
    conf=config()
    heightborder= conf.heightborder
    widthborder = conf.widthborder
    del conf
    if height<heightborder and width<widthborder:#注意这里是小于哦
        if height>=0 and width>=0:#点击位置
            return 1
    return 0

def removedir():
    conf=config()
    removeimgs(conf.lowlocation)
    removeimgs(conf.highlocation)

def removeimgs(dirname):
    if os.path.isdir(dirname):
        try:
            shutil.rmtree(dirname)
        except  Exception as e:
            print(e)
    os.mkdir(dirname)

def transform():
    datadir=conf.datalocation
    flagspm =""
    processim = np.zeros([conf.heightborder, conf.widthborder], dtype=int)
    count=0
    try:
        _,startdelay,enddelay=argv#获取上一个脚本得到的命令行参数
    except  Exception as e:
        startdelay='30'
        enddelay='0'
    startdate=(datetime.datetime.now() + datetime.timedelta(days=eval(startdelay)*-1)).strftime("%Y%m%d")
    enddate = (datetime.datetime.now() + datetime.timedelta(days=eval(enddelay)*-1)).strftime("%Y%m%d")
    i=0
    with open(datadir) as f:
        line = f.readline()
        while line:
            temp= re.split(r"\t", line)
            spm=temp[0];slideendx=eval(temp[1]);slideendy=eval(temp[2]); entiscnx=eval(temp[3]); entiscny=eval(temp[4]); entitywidth=eval(temp[5]); entityheight=eval(temp[6])
            if judegzero(entitywidth, entityheight):
                x = int((slideendx -entiscnx) /entitywidth * conf.widthborder)  # 计算x方向相对坐标
                y = int((slideendy-entiscny) /  entityheight * conf.heightborder)  # 计算y方向的相对坐标
                if judgeoutborder(x, y):  # 首先过滤不需要的点,符合条件的才加以考虑
                    if (flagspm == spm):  # 和原来SPM相同，那么在该图片添加一个像素点
                        count = count + 1
                        processim[y, x] = processim[y, x] + 1
                    else:#和前面不是同一个spm，则保存前一个SPM生成的值为图片，并且准备生成后一个SPM产生的图片
                        if (count >= 1):
                            maxcount = np.max(processim)
                            csv_operator.saveexecl(flagspm, maxcount, count)
                        if (count >= conf.lowthreshold and count<conf.hightthreshold ):
                            new_path = conf.lowlocation +"/"+flagspm + startdate + enddate + ".png"
                            scipy.misc.imsave(new_path, processim)
                        if ( count>=conf.hightthreshold ):
                            new_path = conf.highlocation +"/"+flagspm + startdate + enddate + ".png"
                            scipy.misc.imsave(new_path, processim)
                        flagspm = spm  # 更新当前的flagspm

                        processim = np.zeros([conf.heightborder, conf.widthborder], dtype=int)  # 新申请一个图片
                        processim[y, x] = processim[y, x] + 1
                        count = 1
            line = f.readline()#读取下一条记录
            i=i+1
            if i%10000==0:
                print("当前处理进度：",i)
        try:
            if (flagspm == spm):#这一种情况，就是最后一个家伙
                if (count >= 1):
                    maxcount = np.max(processim)
                    csv_operator.saveexecl(spm, maxcount, count)
                if (count >= conf.lowthreshold and count < conf.hightthreshold):
                    new_path = conf.lowlocation + "/" + flagspm + startdate + enddate + ".png"
                    scipy.misc.imsave(new_path, processim)
                if (count >= conf.hightthreshold):
                    new_path = conf.highlocation + "/" + flagspm + startdate + enddate + ".png"
                    scipy.misc.imsave(new_path, processim)
        except:
            pass
if __name__ == '__main__':
    removedir()#清理之前生成的图片
    transform()#生成新的图片
