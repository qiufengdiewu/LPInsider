# coding=utf-8
#将../目录下的正负样本处理为textCNN可以接受的格式
import pandas as pd

pos_sample=pd.read_csv("../pos_sample.txt",sep="\t",header=None)
neg_sample=pd.read_csv("../neg_sample.txt",sep="\t",header=None)
pos_sample_length=len(pos_sample)
neg_sample_length=len(neg_sample)
f = open("pos_sample_textCNN.txt","w",encoding='utf-8')
for i in range(pos_sample_length):
    f.write(str(pos_sample[3][i])+"\n")
f.close()

f = open("neg_sample_textCNN.txt","w",encoding='utf-8')
for i in range(neg_sample_length):
    f.write(str(neg_sample[2][i])+"\n")
f.close()