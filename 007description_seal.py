#coding:utf-8
import numpy as np
import pandas as pd

#读入数据
pos=pd.read_csv('./in/023pos_sample.txt',sep='\t',header=None)###########
neg=pd.read_csv('./in/023neg_sample.txt',sep='\t',header=None)#######################

posD_length=len(pos)
negD_length=len(neg)

#合并neg和pos

f=open("./out/007X_with_entity_and_stanford_parser.txt","w")#########################################
for i in range(posD_length):#######################
    f.write(str(pos[0][i])+'\t'+str(pos[1][i])+'\t'+str(pos[2][i])+'\n')
for i in range(negD_length):#######################
    f.write(str(neg[0][i])+'\t'+str(neg[1][i])+'\t'+str(neg[2][i])+'\n')

f.close()

#构造对应的标签数组
table=np.append((np.ones(int(posD_length))),(np.zeros(int(negD_length))),axis=0)
np.save('./out/007X_with_entity_and_stanford_parser', table)###############################################

print ("success!")