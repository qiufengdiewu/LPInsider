# coding=utf-8

from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pandas as pd

sentences=[]
POS_of_words=pd.read_csv("./in/023pos_sample_word_transform_to_POS.txt",sep='\t',header=None)
for i in range(len(POS_of_words)):
    description_POS=str(POS_of_words[2][i])
    description_POS=description_POS.split(" ")
    words = []
    for pos in description_POS:
        words.append(pos)
    sentences.append(words)

POS_of_words=pd.read_csv("./in/023neg_sample_word_transform_to_POS.txt",sep='\t',header=None)
for i in range(len(POS_of_words)):
    description_POS=str(POS_of_words[2][i])
    description_POS=description_POS.split(" ")
    words=[]
    for pos in description_POS:
        words.append(pos)
    sentences.append(words)


#附加上所有的词性
words=[]
POSs=pd.read_csv("./in/POSs.txt",sep='\t',header=None)
for i in range(len(POSs)):
    words.append(POSs[0][i])

sentences.append(words)

model=word2vec.Word2Vec(sentences=sentences,min_count=1,size=200)

try:
    print (model['NNS'])
except:
    print ("NNS not in model")

model.save("./out/024POS_of_words.model")