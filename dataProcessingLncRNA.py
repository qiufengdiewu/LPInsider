# -*- coding: utf-8 -*-
import pandas as pd
import string
import re
import numpy as np
import MySQLdb as mySQLDB
'''
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
'''
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

def get_stopwords():
    stopwords = []
    with open('../in/stopwords.txt','r') as f:
        for i in f.readlines():
            stopwords.append(i.strip())
    return stopwords

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    #print("rest"+str(type(res)))
    #print("res:"+str(res))
    return res

def main():
    data = pd.read_csv('../data.txt',sep = '\t',header=None)
    
    stopwords = get_stopwords()
    num = ('1','2','3','4','5','6','7','8','9','0')
    del_ch = string.punctuation
    del_ch = del_ch.replace('-','')
    del_ch = del_ch.replace('/','')

    # 打开数据库连接
    db = mySQLDB.connect(host='127.0.0.1', user='root', passwd='11223366', db='ncrna', charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL查询语句
    sql = "select id from lncrna_abstract"
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        idList = cursor.fetchall()
    except:
        print("Error")
    length_IdList=len(idList)

    for i in range(0,length_IdList):
        sqlFindAbstractByID = "select abstract from lncrna_abstract where id="
        sqlFindAbstractByID+=str(idList[i][0])
        try:
            # 执行SQL语句
            cursor.execute(sqlFindAbstractByID)
            # 获取所有记录列表
            abstract = cursor.fetchone()
        except:
            print("Error.can't find Abstract")
        s=str(abstract[0])
        if(type(s)==float and np.isnan(s)):
            continue
        s=s.lower()
        s = s.lower()  # 全部转成小写字母
        s = re.sub(r'[^\x00-\x7f]', ' ', s)  # 去除非ASCII
        s = s.replace('/', ' ')
        s = lemmatize_sentence(s)
        for j in del_ch:  # 去除标点符号
            while j in s:
                s.remove(j)
        for j in stopwords:  # 去除常用词
            while j in s:
                s.remove(j)
        for j in num:
            while j in s:
                s.remove(j)
        message=""
        for j in s:
            message += str(j)
            message += " "
        #print(message)

        sqlUpdateAbstract="update lncrna_abstract set abstract='"+str(message)+"' where id="+str(idList[i][0])
        #print(sqlUpdateAbstract)
        try:
            #执行sql语句
            cursor.execute(sqlUpdateAbstract)
            #提交到数据库执行
            db.commit()
            #print(sqlUpdateAbstract)
        except :
            #发生错误时候回滚
            db.rollback()
            print("错误提交")
    db.close()

if __name__ == "__main__":
    main()