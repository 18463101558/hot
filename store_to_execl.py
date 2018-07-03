# coding=utf-8
from conf.config import config
import csv
import os
class csv_operator:
    filename = config().clicksaveexecl
    if os.path.isfile(filename):
        try:
            os.remove(filename)
        except  Exception as e:
            print(e)
    csvFile = open( filename,"w+", newline='')
    writer = csv.writer(csvFile)
    writer.writerow(["spm", "maxclick", "totalclick"])
    csvFile.close()
    def saveexecl(spm,maxclickcount,totalcount):
        filename = config().clicksaveexecl
        csvFile = open( filename,"a+", newline='')
        writer = csv.writer(csvFile)
        writer.writerow([spm, maxclickcount, totalcount])
        csvFile.close()
    # def produce_label_execl(new_path):
    #     filename = config().labelexecl
    #     if os.path.isfile(filename):
    #         try:
    #             os.remove(filename)
    #         except  Exception as e:
    #             print(e)
    #     csvFile = open(filename, "w+", newline='')
    #     writer = csv.writer(csvFile)
    #     writer.writerow(["spm", "maxclick", "totalclick"])
    #     csvFile.close()
    def __del__(self):
        try:
            self.csvFile.close()
        except  Exception as e:
            pass