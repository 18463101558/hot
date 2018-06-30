import pymysql.cursors
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
              try:
                  self.releaseconnection(cursor,connection)
              except:
                  pass

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