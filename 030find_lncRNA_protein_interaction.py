# coding=utf-8
import pandas as pd
import requests

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


# 处理xml函数，将其中配套的<.></.>删除，并将<符号删除
def gets(abxml):
    a = abxml.find(b'<')
    b = abxml.find(b'>')
    tag = abxml[a:(b + 1)]
    abxml = abxml.replace(tag,b'')
    abxml = abxml.replace(b'</ArticleTitle>', b'')
    abxml = abxml.replace(b'</AbstractText>', b'')
    while (abxml.find(b'<') != -1):
        a = abxml.find(b'<')
        b = abxml.find(b'>')
        if (b == -1):
            abxml = abxml[:a] + abxml[(a + 1):]
            continue
        tag = abxml[a:(b + 1)]
        try:
            tag1 = tag[0] + b'/' + tag[1:]
        except IndexError:
            print(abxml, b'tag=', tag)
            quit()
        if tag1 in abxml:
            abxml = abxml.replace(tag, b'')
            abxml = abxml.replace(tag1, b'')
        else:
            abxml = abxml[:a] + abxml[(a + 1):]
    ab = abxml.replace(b'\n', b'')
    ab = str(ab.strip())

    return ab[2:len(ab)-1]   #使用python3写会多处前两个字符b'和最后一个字符'


raid=pd.read_csv("./out/029raid2entities_with_reference.txt",sep="\t",header=None)
raid_length=len(raid)
f=open("./out/030raid_reference_abstract.txt","w")
n=int(raid_length/200)

for i in range(n+1):
    pids=""
    n1=200
    if i==n:
        n1=raid_length%200
    for j in range(n1):
        pids=pids+str(raid[11][j])+','
    pids=pids.replace("//",",")

    #接下来就是下载平PMID对应的摘要了。
    # 使用efetch获取文章
    retmode = "xml"
    res = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=" + pids + "&retmode=" + retmode)
    tree = ET.fromstring(res.text.encode('utf-8'))
    tree = ET.ElementTree(tree)
    root = tree.getroot()
    for node in root:
        # 获取摘要
        string = ''
        a = node.find(".//Abstract")
        if a != None:
            # continue
            for elem in a.iter():
                if elem.tag == 'AbstractText':
                    abxml = ET.tostring(elem)
                    ab = gets(abxml)
                    string = string + str(ab) + ' '
        string = string.replace('\t', ' ').strip()
        #此处开始依次往文件中写入RAID、lncRNA、protein、pmid、abstract
        # pid
        pid = node.find(".//PMID").text
        # title
        t0 = node.find(".//ArticleTitle")
        txml = ET.tostring(t0)
        title = str(gets(txml))
        f.write(pid + "\t"+title+"\t"+string+"\t\n")

f.close()