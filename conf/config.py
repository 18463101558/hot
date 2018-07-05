import configparser

class config:
    def __init__(self):
        conf = configparser.ConfigParser()
        conf.read("conf/hotmap.conf")
        self.host = conf.get('connectdb', 'host')#连接数据库的主机
        self.port = eval(conf.get('connectdb', 'port'))  # port是一个数字，所以需要强制转换
        self.user = conf.get('connectdb', 'user')#数据库用户名
        self.password = conf.get('connectdb', 'password')#数据库密码
        self.databasename = conf.get('connectdb', 'databasename')#数据库名称
        self.tablename = conf.get('connectdb', 'tablename')#数据库表名

        self.heightborder = eval(conf.get('border', 'height'))# 生成的图片高度
        self.widthborder = eval(conf.get('border', 'width'))#生成图片的宽度

        self.clicksaveexecl=conf.get('execl','clicksaveexecl')#保存最大点击量，spm等点击信息的execl存放路径

        self.starttime= eval(conf.get('sampledate', 'starttime'))#采样的数据，是从今天的前几天开始取
        self.endtime = eval(conf.get('sampledate', 'endtime'))#采样数据的截至日期，例如start=-1，end=0就表示只取昨天和今天的数据

        self.batch_size=eval(conf.get('resnet','batch_size'))#训练的batch_size大小
        self.epochs= eval(conf.get('resnet', 'epochs'))#训练的迭代次数
        self.learning_rate= eval(conf.get('resnet', 'learning_rate'))#初始状态的学习率
        self.model_path =conf.get('resnet', 'model_path')#训练好的模型的保存路径
        self.save_frequency = eval(conf.get('resnet', 'save_frequency'))  # 训练好的模型的保存路径
        self.monitoring_rate= eval(conf.get('resnet', 'monitoring_rate'))
        self.revert_flag = eval(conf.get('resnet', 'revert_flag'))
        self.modelpath=conf.get('resnet', 'modelpath')
