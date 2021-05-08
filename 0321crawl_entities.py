# coding=utf-8
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import time
import random

def crawl(raID):
    request=urllib.request.Request("http://www.rna-society.org/raid2/php/more.php?raid="+raID)
    response=urllib.request.urlopen(request)
    html=response.read().decode("utf-8")
    soup = BeautifulSoup(html, 'html.parser')
    tag_soup = soup.find_all(id="detailContainer")
    flag_Alias=False
    detailContainer_length=len(tag_soup)

    for i in range(detailContainer_length):
        context=tag_soup[i]
        context_find=context.find_all("th")
        for con_th in context_find:
            str_con_th=str(con_th).lower()

            if str_con_th.find("Symbol".lower())>=0:
                siblings = con_th.find_next_siblings()
                symbol1 = str(siblings[0].string)
                symbol2 = str(siblings[1].string)
            if str_con_th.find("Category".lower())>=0:
                siblings = con_th.find_next_siblings()
                Category1 = str(siblings[0].string).lower()
                Category2 = str(siblings[1].string).lower()


            if str_con_th.find("Alias".lower())>=0:#Alias在rna protein symbol之后
                    siblings=con_th.find_next_siblings()
                    symbol1_Alias = str(siblings[0].string)
                    symbol2_Alias = str(siblings[1].string)

                    if Category1=="lncrna":
                        f_entities = open("./out/0321crawl_lncRNA.txt", "a+", encoding='utf-8')
                        f_entities.write( raID + "\t" + symbol1 + "\t" + symbol1_Alias+'\n')
                        f_entities.close()
                        print("lncrna\t"+ raID + "\t" + symbol1 + "\t" + symbol1_Alias)
                        flag_Alias = True
                    elif Category1=="protein":
                        f_entities = open("./out/0321crawl_protein.txt", "a+", encoding='utf-8')
                        f_entities.write(raID + "\t" + symbol1 + "\t" + symbol1_Alias + "\n")
                        f_entities.close()
                        print("protein\t"+raID + "\t" + symbol1 + "\t" + symbol1_Alias )
                        flag_Alias=True

                    if Category2=="lncrna":
                        f_entities = open("./out/0321crawl_lncRNA.txt", "a+", encoding='utf-8')
                        f_entities.write(raID + "\t" + symbol2 + "\t" + symbol2_Alias + '\n')
                        f_entities.close()
                        print("lncrna\t"+raID + "\t" + symbol2 + "\t" + symbol2_Alias )
                        flag_Alias = True
                    elif Category2=="protein":
                        f_entities = open("./out/0321crawl_protein.txt", "a+", encoding='utf-8')
                        f_entities.write(raID + "\t" + symbol2 + "\t" + symbol2_Alias + "\n")
                        f_entities.close()
                        print("protein\t"+raID + "\t" + symbol2 + "\t" + symbol2_Alias )
                        flag_Alias=True

    if flag_Alias == False:
        f_entities_not_writed = open("./out/0321entity_not_writed.txt", "a+",encoding='utf-8')
        f_entities_not_writed.write(str(raID)+"\n")
        f_entities_not_writed.close()
    return raID

raid_protein=pd.read_csv("./out/032raid_proteins_unique.txt",sep="\t",header=None)
raid_lncRNA=pd.read_csv("./out/032raid_lncRNAs_unique.txt",sep="\t",header=None)

for i in range(len(raid_protein)):
    try:
        RAID = str(raid_protein[0][i])
        print(str(i)+"\t"+crawl(RAID))
        if i%100==0 and i!=0:
            print("睡眠ing")
            time.sleep(random.randint(30,60))
    except:
        f_exception_RAID = open("./out/0321exception_protein_RAID.txt", "a+",encoding='utf-8')
        f_exception_RAID.write(str(RAID)+"\tprotein\n")
        f_exception_RAID.close()

for i in range(103,len(raid_lncRNA)):
    try:
        RAID = str(raid_lncRNA[0][i])
        print(str(i) + "\t" + crawl(RAID))
        if i%100==0 and i!=0:
            print("睡眠ing")
            time.sleep(random.randint(30,60))
    except:
        f_exception_RAID = open("./out/0321exception_lncRNA_RAID.txt", "a+", encoding='utf-8')
        f_exception_RAID.write(str(RAID) + "\tlncRNA\n")
        f_exception_RAID.close()
