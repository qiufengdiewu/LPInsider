# coding=utf-8
#当代码执行完毕之后，需要对生成的pos_sample.txt和neg_sample.txt进行信息核对
import pandas as pd
pos=pd.read_csv("./in/pos_sample.txt",sep="\t",header=None)
neg=pd.read_csv("./in/neg_sample.txt",sep="\t",header=None)
pos_length=len(pos)
neg_length=len(neg)
f_pos=open("./out/pos.txt","w")
f_neg=open("./out/neg.txt","w")
pos_not_writed=[]
for i in range(pos_length):
    lncRNA=str(pos[0][i])
    protein=str(pos[1][i])
    relation=str(pos[2][i])
    description=str(pos[3][i])
    find_lncRNA=lncRNA in description
    find_protein=protein in description
    if find_lncRNA and find_protein:
        description=description.replace(lncRNA,"<lncRNA>"+lncRNA+"</lncRNA>")
        description=description.replace(protein,"<protein>"+protein+"</protein>")
        f_pos.write(lncRNA+"\t"+protein+"\t"+relation+"\t"+description+"\n")
    else:
        pos_not_writed.append(i)
print(pos_not_writed)
f_pos.close()

neg_not_writed=[]
for i in range(neg_length):
    lncRNA=str(neg[0][i])
    protein=str(neg[1][i])
    description=str(neg[2][i])
    find_lncRNA = lncRNA in description
    find_protein = protein in description
    if find_lncRNA and find_protein:
        description = description.replace(lncRNA, "<lncRNA>" + lncRNA + "</lncRNA>")
        description = description.replace(protein, "<protein>" + protein + "</protein>")
        f_neg.write(lncRNA+"\t"+protein+"\t"+description+"\n")
    else:
        neg_not_writed.append(i)
print(neg_not_writed)
f_neg.close()

