# coding=utf-8
import pandas as pd

POS_classified = pd.read_csv("./in/POS_classified.txt", sep='\t', header=None)
length_classified = len(POS_classified)
POS_classified0 = POS_classified[0]
POS_classified1 = POS_classified[1]


def united(word):
    for i in range(length_classified - 1):
        flag = 0
        temp_word = ""
        temp_POS = str(POS_classified1[i]).split(" ")
        for j in range(len(temp_POS)):
            if word == temp_POS[j]:
                flag = 1
                temp_word = POS_classified0[i]
                break
            else:
                temp_word = POS_classified0[length_classified - 1]
        if flag == 1:
            break
    return temp_word


POS = pd.read_csv("./out/0261abstract2POS.txt", sep='\t', header=None)
f = open("./out/0262abstract2POS_united.txt", "w")

for i in range(len(POS)):
    title = POS[0][i]
    title = str(title).split(" ")
    temp_title = ""
    for j in range(len(title)):
        if title[j]!="":
            temp_title += united(str(title[j]))
            if j < (len(title) - 1):
                temp_title += " "
    f.write(temp_title + "\t")

    abstract = POS[1][i]
    temp_abstract = ""
    if type(abstract) != type(0.1):
        abstract = str(abstract).split(" ")
        for j in range(len(abstract)):
            if abstract[j]!="":
                temp_abstract += united(str(abstract[j]))
                if j < (len(abstract) - 1):
                    temp_abstract += " "
    f.write(temp_abstract + "\n")

f.close()
