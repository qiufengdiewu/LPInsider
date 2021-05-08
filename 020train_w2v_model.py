# coding=utf-8
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

f = open("./out/020train_w2v_temp_data.txt","w")

with open("017in_w2v_pos.txt","r") as pos_txt:
    line=pos_txt.readline()
    while line:
        line=pos_txt.readline()
        description=str(line)
        description = description.replace('(', ' ')
        description = description.replace(',', ' ')
        description = description.replace(')', ' ')
        description = description.replace('.', ' ')
        description = description.replace("'", ' ')
        description = description.replace(':', ' ')
        description = description.replace('[', ' ')
        description = description.replace(']', ' ')
        description = description.replace('/', ' ')
        f.write(str(description) + '\n')

with open("017in_w2v_neg.txt","r") as neg_txt:
    line=neg_txt.readline()
    while line:
        line=neg_txt.readline()
        description=str(line)
        description = description.replace('(', ' ')
        description = description.replace(',', ' ')
        description = description.replace(')', ' ')
        description = description.replace('.', ' ')
        description = description.replace("'", ' ')
        description = description.replace(':', ' ')
        description = description.replace('[', ' ')
        description = description.replace(']', ' ')
        description = description.replace('/', ' ')
        f.write(str(description) + '\n')
f.close()

temp_txt=open("./out/020train_w2v_temp_data.txt","r")
model = word2vec.Word2Vec(LineSentence(temp_txt),min_count=1,size=200)
print(model)
model.save('./out/020Word2vec_model')


print(model['methylation'])
