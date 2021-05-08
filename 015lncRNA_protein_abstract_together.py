# -*- coding: utf-8 -*-
import MySQLdb as mySQLDB
import requests


try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


# 处理xml函数，将其中配套的<.></.>删除，并将<符号删除
def gets(abxml):
    a = abxml.find('<')
    b = abxml.find('>')
    tag = abxml[a:(b + 1)]
    abxml = abxml.replace(tag, '')
    abxml = abxml.replace('</ArticleTitle>', '')
    abxml = abxml.replace('</AbstractText>', '')
    while (abxml.find('<') != -1):
        a = abxml.find('<')
        b = abxml.find('>')
        if (b == -1):
            abxml = abxml[:a] + abxml[(a + 1):]
            continue
        tag = abxml[a:(b + 1)]
        try:
            tag1 = tag[0] + '/' + tag[1:]
        except IndexError:
            print (abxml, 'tag=', tag)
            quit()
        if tag1 in abxml:
            abxml = abxml.replace(tag, '')
            abxml = abxml.replace(tag1, '')
        else:
            abxml = abxml[:a] + abxml[(a + 1):]
    ab = abxml.replace('\n', '')
    ab = ab.strip()

    return ab



def saveData():

    f=open("./out/015lncRNA.txt","w")
    # 打开数据库连接
    db = mySQLDB.connect(host='127.0.0.1', user='root', passwd='11223366', db='ppi_corpus', charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL查询语句
    sql = "select pmid from lncrna_protein_together;"  # 获取PID的记录条数总数
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        pmid_list = cursor.fetchall()
        print(pmid_list)
    except:
        print("Error")

    number = len(pmid_list)
    # 防止过多报错，每次获取100篇
    n = number / 200
    print(number)
    for i in xrange(n + 1):
        pids = ''
        n1 = 200
        if (i == n):
            n1 = number % 200
        for j in xrange(n1):
            pids = pids + str(pmid_list[i * 200 + j][0]) + ','
        pids.strip(',')
        print(pids)
        # 使用efetch获取文章
        retmode = "xml"
        res = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=" + pids + "&retmode=" + retmode)
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
            abstractText = abstractText.strip("'")
            # pid
            pmid = node.find(".//PMID").text
            # year
            year = node.find(".//PubDate/").text
            year = year[:4]
            # title
            title = ''
            t0 = node.find(".//ArticleTitle")
            txml = ET.tostring(t0)
            t0 = gets(txml)
            title = title + t0
            #####在这里对title和year进行处理
            #title = dataprocessing(title)
            #abstractText = dataprocessing(abstractText)
            linetxt=str(pmid)+"\t"+str(year)+"\ttitle\t"+str(title)+"\tabstract\t"+str(abstractText)+"\t\n"
            f.write(linetxt)

        print("loading,---" + str((i + 1) * 200) + "/" + str(number) + '---')
    db.close()
    f.close()

if __name__ == "__main__":
    saveData()
