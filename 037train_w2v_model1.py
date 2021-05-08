# coding=utf-8

from gensim.models import word2vec
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

neg_sample = pd.read_csv('./in/023neg_sample_preprocess.txt',sep='\t',header=None)
pos_sample = pd.read_csv('./in/023pos_sample_preprocess.txt',sep='\t',header=None)
sentences = []
for i in range(len(pos_sample)):
    pos_sample_text = str(pos_sample[2][i])
    temp_words = pos_sample_text.split(" ")
    words = []
    for word in temp_words:
        if len(word) > 0:
           words.append(word)
    sentences.append(words)

for i in range(len(neg_sample)):
    neg_sample_text = str(neg_sample[2][i])
    temp_words = neg_sample_text.split(" ")
    words = []
    for word in temp_words:
        if len(word) > 0:
            words.append(word)
    sentences.append(words)

model=word2vec.Word2Vec(sentences,min_count=1,size=200)
print (model)
model.save('./out/037train_w2v_model')
print(model["Knockdown"])

