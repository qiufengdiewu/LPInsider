# -*- coding: UTF-8 -*-
from Bio import Entrez
import MySQLdb as mySQLDB
Entrez.email="A.N.Other@example.com"
"""
def savePID():
    '''
        ①savePID()摘录的PMID的写法在最下方的注释部分。
        ②摘录了来自RAID的
    '''
    f=open("./out/999lncRNA.txt","r")
    lncRNA_set=f.readline()
    lncRNAs=lncRNA_set.split(" ")

    returnCount=100000#每次可以最大返回十万条数据。
    for lncRNA in lncRNAs:
        handle = Entrez.esearch(db="pubmed", term=str(lncRNA), RetMax=returnCount)
        '''
            returnCount目前是够用的，但是不能保证以后一定可以。如果运行错误，则参照官网给出的函数参数进行修改
        '''
        record=Entrez.read(handle)
        print(lncRNA)
        print(record)
        idList=record["IdList"]
        count=record["Count"]
                #打开数据库连接
        db = mySQLDB.connect(host='127.0.0.1',user='root',passwd='11223366',db='ncrna',charset='utf8')
        #使用cursor()方法获取操作游标
        cursor=db.cursor()
        for i in range(0,int(count)):
            sql = "insert into lncrna_pid (pid) values(" + idList[i] + ")"
            try:
                #执行SQL语句
                cursor.execute(sql)
                #提交到数据库执行
                db.commit()
            except:
                db.rollback()
                #print("Error,can't insert data.  "+str(sql))
        db.close()
"""

#①上面的savePID先注释，并执行这一部分。注释这一部分后在执行上面的savePID()
def savePID():
    
    returnCount=100000#每次可以最大返回十万条数据。
    #handle=Entrez.esearch(db="pubmed",term="lncRNA",RetMax=returnCount)
    handle = Entrez.esearch(db="pubmed", term="lncRNA", RetMax=returnCount)
    '''
        这些参数值目前是够用的，但是不能保证以后一定可以。如果运行错误，则参照官网给出的函数参数进行修改
    '''
    record=Entrez.read(handle)
    print(record)
    idList=record["IdList"]
    count=record["Count"]
    print("Count"+count)

    #打开数据库连接
    db = mySQLDB.connect(host='127.0.0.1',user='root',passwd='11223366',db='ncrna',charset='utf8')
    #使用cursor()方法获取操作游标
    cursor=db.cursor()
    for i in range(0,int(count)):
        sql = "insert into lncrna_pid (pid) values(" + idList[i] + ")"
        try:
            #执行SQL语句
            cursor.execute(sql)
            #提交到数据库执行
            db.commit()
            print(sql)
        except:
            db.rollback()
            #print("Error,can't insert data.  "+str(sql))
    db.close()


if __name__ == "__main__":
    savePID()