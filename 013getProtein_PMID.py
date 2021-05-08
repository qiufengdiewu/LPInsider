#!/usr/bin/python
# -*- coding: UTF-8 -*-
from Bio import Entrez
import MySQLdb as mySQLDB
Entrez.email="A.N.Other@example.com"

def savePID():
    returnCount = 100000  # 每次可以最大返回十万条数据。2019/6/26返回的最大数量是6918331
    handle=Entrez.esearch(db="pubmed",term="protein")
    record = Entrez.read(handle)
    print(record)
    count=int(record["Count"])
    print("Count" + str(count))
    traversal_count=int(count/100000)+1
    print (traversal_count)
    for j in range(0,traversal_count):
        return_start=returnCount*j
        print("return_start"+str(return_start))
        handle=Entrez.esearch(db="pubmed",term="protein",retstart=return_start,RetMax=returnCount)
        record = Entrez.read(handle)
        #print(record)
        idList = record["IdList"]

        # 打开数据库连接
        db = mySQLDB.connect(host='127.0.0.1', user='root', passwd='11223366', db='ppi_corpus', charset='utf8')
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        for i in range(0, int(returnCount)):#在最后一次插入数据库的时候会出现异常。但是依旧可以得到所有的PMID
            sql = "insert into protein_pmid (pmid) values(" + idList[i] + ")"
            try:
                # 执行SQL语句
                cursor.execute(sql)
                # 提交到数据库执行
                db.commit()
                # print(sql)
                # 这些语句执行之后不能保证PID的唯一，我给的解决方案是讲数据库中的PID设置为unique来避免这个问题
                '''SQL语句如下：
                CREATE TABLE `lncrna_pid` ( `Id` int(11) NOT NULL AUTO_INCREMENT, `pid` int(11) NOT NULL DEFAULT '0',  PRIMARY KEY (`Id`),  UNIQUE KEY `pid` (`pid`)) ENGINE=InnoDB AUTO_INCREMENT=15374 DEFAULT CHARSET=utf8;'''
            except:
                # Rollback in case there is any error
                db.rollback()
                print("Error,can't insert data.  " + str(sql))

    db.close()

if __name__ == "__main__":
    savePID()