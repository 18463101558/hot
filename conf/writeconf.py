# coding=utf-8
import configparser
import sys

def writeConfig(filename):
    config = configparser.ConfigParser()
    # set db
    section_name = 'connectdb'
    config.add_section(section_name)
    config.set(section_name, 'driver', 'pymysql')
    config.set(section_name, 'host', '10.126.83.92')
    config.set(section_name, 'port', '3306')
    config.set(section_name, 'user', 'readuser')
    config.set(section_name, 'password', '123456')
    config.set(section_name, 'databasename', 'featuredb')
    config.set(section_name, 'tablename', 'featuredb.anti_mouse_clickdata_detail_m_whole')

    section_name = 'border'
    config.add_section(section_name)
    config.set(section_name, 'width', '1365')
    config.set(section_name, 'height', '143')
    # write to file
    config.write(open(filename, 'a'))

# def updateConfig(filename, section, **keyv):
#     config = ConfigParser.ConfigParser()
#     config.read(filename)
#     print
#     config.sections()
#     for section in config.sections():
#         print
#         "[", section, "]"
#         items = config.items(section)
#         for item in items:
#             print
#             "\t", item[0], " = ", item[1]
#     print
#     config.has_option("dbname", "MySQL")
#     print
#     config.set("db", "dbname", "11")
#     print
#     "..............."
#     for key in keyv:
#         print
#         "\t", key, " = ", keyv[key]
#     config.write(open(filename, 'r+'))


if __name__ == '__main__':
    file_name = 'hotmap.conf'
    writeConfig(file_name)
    print("写入完毕")
