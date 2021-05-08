# -*- coding: utf-8 -*-
import MySQLdb as mySQLDB
import requests
import traceback
import re
import string

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from Bio import Entrez
Entrez.email="A.N.Other@example.com"


#处理xml函数，将其中配套的<.></.>删除，并将<符号删除
def gets(abxml):
    a = abxml.find('<')
    b = abxml.find('>')
    tag = abxml[a:(b+1)]
    abxml = abxml.replace(tag,'')
    abxml = abxml.replace('</ArticleTitle>','')
    abxml = abxml.replace('</AbstractText>','')
    while(abxml.find('<')!=-1):
        a = abxml.find('<')
        b = abxml.find('>')
        if(b == -1):
            abxml = abxml[:a]+abxml[(a+1):]
            continue
        tag = abxml[a:(b+1)]
        try:
            tag1 = tag[0] + '/' + tag[1:]
        except IndexError:
            print (abxml,'tag=',tag)
            quit()
        if tag1 in abxml:
            abxml = abxml.replace(tag,'')
            abxml = abxml.replace(tag1,'')
        else :
            abxml = abxml[:a]+abxml[(a+1):]
    ab = abxml.replace('\n','')
    ab = ab.strip()
    
    return ab

def get_stopwords():
    stopwords = []
    with open('../in/stopwords.txt','r') as f:
        for i in f.readlines():
            stopwords.append(i.strip())
    return stopwords

def dataprocessing(s):
    s=s.lower()
    s = re.sub(r'[^\x00-\x7f]', ' ', s)  # 去除非ASCII
    temp = s.decode("utf8")
    s = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), " ".decode("utf8"), temp)
    s = s.replace('-', ' ')
    s = s.replace('/',' ')
    stopwords = get_stopwords()
    for j in s:#去除常用词
        if j in stopwords:
            s.replace(j,"")

    del_ch=string.punctuation#去除所有标点符号
    del_ch=del_ch.replace('-','')
    del_ch = del_ch.replace('/', '')
    for  j in s:
        if j in del_ch:
            s.replace(j,"")


    num = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')
    for j in s:
        if j in num:
            s.replace(j,"")
    return s

f=open("./out/lncRNA_abstract.txt","w")

def saveData():
    failCount=0
    #f = open('failInsertIntoDatabaseAboutLncrna.txt', 'w')
    # 打开数据库连接
    db = mySQLDB.connect(host='127.0.0.1', user='root', passwd='11223366', db='ncrna', charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL查询语句
    sql = "select pid from lncrna_pid"  # 获取PID的记录条数总数
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        idlist = cursor.fetchall()
        print(idlist)
    except:
        print("Error")

    number = len(idlist)
    #防止过多报错，每次获取100篇
    n = int(number / 200)
    print(number)
    for i in range(n+1):
        pids = ''
        n1 = 200
        if(i == n):
            n1 =int( number % 200)
        for j in range(n1):
            pids = pids + str(idlist[i*200 + j][0] )+ ','
        pids.strip(',')
        print(pids)
        # 使用efetch获取文章
        retmode = "xml"
        res = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=" + pids + "&retmode=" + retmode)
        tree = ET.fromstring(res.text.encode('utf-8'))
        tree = ET.ElementTree(tree)
        root = tree.getroot()
        for node in root:
            # 获取摘要
            abstractText = ''
            a = node.find(".//Abstract")
            if a != None:
                # continue
                for elem in a.iter():
                    if elem.tag == 'AbstractText':
                        abxml = ET.tostring(elem)
                        ab = gets(abxml)
                        abstractText = abstractText + ab + ' '
            abstractText = abstractText.replace('\t', ' ').strip()
            abstractText=abstractText.strip("'")
            # pid
            pid = node.find(".//PMID").text
            # year
            year = node.find(".//PubDate/").text
            year=year[:4]
            # title
            title = ''
            t0 = node.find(".//ArticleTitle")
            txml = ET.tostring(t0)
            t0 = gets(txml)
            title= title + t0
            ##############
            #将pmid，year，title，abstract保存在txt文件中

            message=str(pid)+"\t"+str(year)+"\t"+str(title)+"\t"+abstractText+"\n"
            f.write(message)

            #####在这里对title和year进行处理
            title=dataprocessing(title)
            abstractText=dataprocessing(abstractText)
            sqlInsertRecord="insert into lncRNA_abstract (PID,year,title,abstract) values("+ pid+","+year+",'"+title+"','"+abstractText + "')"

            try:
                # 执行SQL语句
                cursor.execute(sqlInsertRecord)
                # 提交到数据库执行
                db.commit()
            #except Exception, e:
            except:
                traceback.print_exc()
                db.rollback()
                failCount=failCount+1
                print("Error!pmid:"+pid)
        print("loading,---" + str((i + 1) * 200) + "/" + str(number) + '---')
    db.close()

    print(failCount)
if __name__ == "__main__":
    saveData()
    f.close()
'''
导入数据的时候，mysql报错 ERROR 1406 : Data too long for column Data too long for column
解决办法:

在my.ini里找到
sql-mode=”STRICT_TRANS_TABLES,NO_AUTO_Create_USER,NO_ENGINE_SUBSTITUTION”
把其中的STRICT_TRANS_TABLES,去掉,
或者把sql-mode=STRICT_TRANS_TABLES,NO_AUTO_Create_USER,NO_ENGINE_SUBSTITUTION 
MySQL数据库的my.ini文件被我修改了
字段中不可以出现MySQL数据库预留的符号，否则会出错。
由于MySQL对语法要求较高，所以处理这样的问题有两种思路
    一：改用txt或者cvs文件
    二：直接对文件进行处理
'''