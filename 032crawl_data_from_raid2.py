# coding=utf-8
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import time
import random

def crawl(raID):
    lncRNA=""
    protein=""
    request=urllib.request.Request("http://www.rna-society.org/raid2/php/more.php?raid="+raID)
    response=urllib.request.urlopen(request)
    html=response.read().decode("utf-8")
    soup = BeautifulSoup(html, 'html.parser')
    tag_soup = soup.find_all(id="detailContainer")
    flag_f_description=False
    flag_Alias=False
    detailContainer_length=len(tag_soup)

    for i in range(detailContainer_length):
        context=tag_soup[i]
        context_find=context.find_all("th")
        for con_th in context_find:
            str_con_th=str(con_th).lower()

            if str_con_th.find("Symbol".lower())>=0:
                siblings = con_th.find_next_siblings()
                raid_lncRNA = str(siblings[0].string)
                raid_protein = str(siblings[1].string)

            if str_con_th.find("Alias".lower())>=0:#Alias在rna protein symbol之后
                siblings=con_th.find_next_siblings()
                lncRNA_Alias = str(siblings[0].string)
                protein_Alias = str(siblings[1].string)
                f_entities = open("./out/032crawl_entities.txt", "a+",encoding='utf-8')
                f_entities.write(
                    raID + "\t" + raid_lncRNA + "\t" + lncRNA_Alias + "\t" + raid_protein + "\t" + protein_Alias + "\t\n")
                f_entities.close()
                print(raID+"\tf_entities has done")
                flag_Alias=True
                lncRNA = raid_lncRNA
                protein = raid_protein
    # 原网站的tag_soup(id=detailContainer)部分的长度不一定。只能遍历查找

    context=tag_soup[detailContainer_length-1]
    context_find= context.find_all("th")
    exist_pmid=False
    for con_th in context_find:
        str_con_th=str(con_th).lower()
        if str_con_th.find("PMID".lower())>=0:
            exist_pmid=True
        if str_con_th.find("description")>=0 and exist_pmid==True:
            sibling = con_th.find_next_sibling()
            description=""+ str(sibling.string)
            f_description = open("./out/032crawl_description.txt", "a+",encoding='utf-8')
            f_description.write(raID + "\t" + lncRNA + "\t" + protein + "\t" + description + "\t\n")
            f_description.close()
            print(raID + "\tf_description has done")
            flag_f_description = True

    if flag_f_description == False:
        f_unconfirm_description = open("./out/032crawl_unconfirm_description.txt", "a+",encoding='utf-8')
        f_unconfirm_description.write(raID + "\t" + lncRNA + "\t" + protein + "\t\n")
        f_unconfirm_description.close()
        print(raID + "\tf_unconfirm_description has done")

    if flag_f_description == False and flag_Alias == False:
        f_entities_not_writed = open("./out/032entity_not_writed.txt", "a+",encoding='utf-8')
        f_entities_not_writed.write(str(raID) + "\t" + lncRNA + "\t" + protein + "\t\n")
        f_entities_not_writed.close()
    return raID

raid=pd.read_csv("I:/RAID_V2/raid2entities.txt",sep="\t",header=None)

for i in range(len(raid)):
    try:
        RAID = str(raid[0][i])
        lncRNA = str(raid[1][i]).lower()
        protein = str(raid[5][i]).lower()
        print(str(i)+"\t"+crawl(RAID))
        if i%100==0 and i!=0:
            print("睡眠ing")
            time.sleep(random.randint(30,60))
    except:
        f_exception_RAID = open("./out/032exception_RAID.txt", "a+",encoding='utf-8')
        f_exception_RAID.write(str(RAID)+"\t\n")
        f_exception_RAID.close()
        print(RAID + "\tf_exception_RAID has done***********************************")
