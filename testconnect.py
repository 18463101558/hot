import pymysql.cursors
import configparser
import time,datetime
import numpy as np
import cv2
import sys
from conf.config import config
class DBUtil(object):
    def __init__(self):
        conf = config()
        self.host=conf.host
        self.port=conf.port#port是一个数字，所以需要强制转换
        self.user=conf.user
        self.password= conf.password
        self.db=conf.databasename
        self.tablename=conf.tablename
        del conf
    def getconnection(self):# 连接MySQL数据库
        connection = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, db=self.db)
        return connection
    def executesearch(self,sql,arg):#根据SQL语句和参数进行查询
        try:
            connection=self.getconnection()
            cursor = connection.cursor()# 通过cursor创建游标
            cursor.execute(sql,arg)#执行SQL语句
            results = cursor.fetchall()# 取出所有的查询结果
            return results
        except  Exception as e:
            print(e)
            return []
        finally:
            self.releaseconnection(cursor,connection)

    def releaseconnection(self,cursor,connection):#无论如何也要尝试关闭游标和数据库连接
            try :
                cursor.close()
            except Exception as e:
                print(e)
            finally:
                try:
                    connection.close()
                except Exception as e:
                    print(e)
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
          +tablename+" where spm=%s and dt>=%s and dt<=%s and touch_type=2 order by pos limit 0,3000; "
    args=(spm,startdate,enddate)
    results =conn.executesearch(sql,args)

    conf = config()
    processim = np.zeros([conf.heightborder,conf.widthborder], dtype=int)#高度和宽度
    count=0
    for data in results:
        if judegzero(data[4],data[5])!=0:
            x =int((data[0]-data[2])/data[4]*conf.widthborder)# 鼠标点击位置减去容器框位置除以容器框的宽度
            y=int((data[1]-data[3])/data[5]*conf.heightborder)#
            if judgeoutborder(x,y):
                count=count+1
                processim[y, x] =processim[y, x]+1
                if count%1000==0:
                    print("处理数据进度：",count)
                    print(str(data))

    maxcount = np.max(processim)
    print("最大点击次数为：",maxcount)
    #processim=processim*50
    processim = processim * 255 / maxcount
    new_path ="imgs/"+startdate+enddate+spm+".png"
    print("总共的点击点为：",count)
    cv2.imwrite(new_path, processim)
def spmlist(startdate,enddate):
    conn = DBUtil()
    conf = config()
    tablename = conf.tablename
    sql = "SELECT distinct spm  FROM " \
          +tablename+" where  dt>=%s and dt<=%s and touch_type=2 order by pos ; "
    args=(startdate,enddate)
    results = conn.executesearch(sql, args)
    return results;
if __name__ == '__main__':
    spm='u-2c13wpanv3v43nkddh1'
    startdate=(datetime.datetime.now() + datetime.timedelta(days=-1)).strftime("%Y%m%d")
    enddate = datetime.datetime.now().strftime("%Y%m%d")
    #producepicture(spm, startdate, enddate)
    results=spmlist(startdate,enddate)
    for spm in results:
        producepicture(spm[0], startdate, enddate)
