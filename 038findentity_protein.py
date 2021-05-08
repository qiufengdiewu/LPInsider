# coding=utf-8
"""
本程序的作用：找到uniprot中的所有蛋白质的名称。
uniprot的文件的中每个protein实体之间使用//符号进行分隔；
我们需要用到的名称有RecName和ORFNames
f1文件的格式如下： ORFNames\n
"""
path = "I:/uniprot_sprot_data/"
path_file = "I:/uniprot_sprot_data/uniprot_sprot.txt"
f = open(file=path_file, mode="r")
f1 = open(file=(path+"proteins_name.txt"), mode="w")

line = f.readline()
while line:
    if line.find("ORFNames") >= 0:
        equ_loc = line.find("=")
        semicolon_loc = line.find(";")
        ORFNames = line[equ_loc+1: semicolon_loc]
        f1.write(ORFNames+"\n")
    line = f.readline()

f.close()
f1.close()
