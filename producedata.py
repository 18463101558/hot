import datetime
import numpy as np
import cv2
import os
import csv
import shutil
from DBUtil import DBUtil
from conf.config import config
from store_to_execl import  csv_operator
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
def producepicture(spm,startdate,enddate):
    conn=DBUtil()
    tablename=conn.tablename
    print("现在处理的spm：",spm)
    sql = "SELECT slideend_x,slideend_y,entity_x,entity_y,entity_width,entity_height  FROM " \
          +tablename+" where spm=%s and dt>=%s and dt<=%s and touch_type=2 order by pos limit 0,100000; "
    args=(spm,startdate,enddate)
    results =conn.executesearch(sql,args)

    conf = config()
    processim = np.zeros([conf.heightborder,conf.widthborder], dtype=int)#高度和宽度
    count=0
    for data in results:
        if judegzero(data[4],data[5])!=0:
            x =int((data[0]-data[2])/data[4]*conf.widthborder)# 鼠标点击位置减去容器框位置除以容器框的宽度
            y=int((data[1]-data[3])/data[5]*conf.heightborder)#360 120 另外一组是
            if judgeoutborder(x,y):
                count=count+1
                #processim[y, x] =255
                processim[y, x] =processim[y, x]+1
                if count%1000==0:
                    print("处理数据进度：",count)
                    print(str(data))

    maxcount = np.max(processim)
    print("最大点击次数为：",maxcount)
    processim = processim * 255 / maxcount
    new_path ="imgs/"+spm+startdate+enddate+".png"
    print("总点击次数为：",count)
    if(count>=1000):
        csv_operator.saveexecl(spm,maxcount,count)
        cv2.imwrite(new_path, processim)
def spmlist(startdate,enddate):
    conn = DBUtil()
    conf = config()
    tablename = conf.tablename
    sql = "SELECT distinct spm  FROM " \
          +tablename+" where  dt>=%s and dt<=%s and touch_type=2  ; "
    # sql = "SELECT dt  FROM " \
    #       +tablename+" where  dt>=%s and dt<=%s order by dt asc limit 1 ; "
    args=(startdate,enddate)
    results = conn.executesearch(sql, args)

    return results
def prework_before_label():
    filename = config().labelexecl
    if os.path.isfile(filename):
        try:
            os.remove(filename)
        except  Exception as e:
            print(e)
def remove():
    if os.path.isdir("imgs"):
        try:
            shutil.rmtree('imgs')
        except  Exception as e:
            print(e)
    os.mkdir("imgs")
if __name__ == '__main__':
    remove()
    ##spm='u-2c13wpanv3v43nkddh1'
    startdate=(datetime.datetime.now() + datetime.timedelta(days=-1)).strftime("%Y%m%d")
    enddate = (datetime.datetime.now() + datetime.timedelta(days=0)).strftime("%Y%m%d")
    #producepicture(spm, startdate, enddate)
    results=spmlist(startdate,enddate)
    spmlist=[]
    for spm in results:
        producepicture(spm[0], startdate, enddate)
