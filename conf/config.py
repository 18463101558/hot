import configparser

class config:
    def __init__(self):
        conf = configparser.ConfigParser()
        conf.read("conf/hotmap.conf")
        self.host = conf.get('connectdb', 'host')
        self.port = eval(conf.get('connectdb', 'port'))  # port是一个数字，所以需要强制转换
        self.user = conf.get('connectdb', 'user')
        self.password = conf.get('connectdb', 'password')
        self.databasename = conf.get('connectdb', 'databasename')
        self.tablename = conf.get('connectdb', 'tablename')

        self.heightborder = eval(conf.get('border', 'height'))
        self.widthborder = eval(conf.get('border', 'width'))

        self.clicksaveexecl=conf.get('execl','clicksaveexecl')
        self.labelexecl= conf.get('execl', 'labelexecl')