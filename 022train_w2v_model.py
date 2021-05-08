# coding=utf-8

from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

temp_txt=open("./out/021train_w2v_preprocess.txt","r")
model=word2vec.Word2Vec(LineSentence(temp_txt),min_count=1,size=200)
print(model)
model.save('./out/022train_w2v_model')
print(model["Cellular"])