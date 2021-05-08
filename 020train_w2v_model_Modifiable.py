# coding=utf-8

from gensim.models import word2vec
import pandas as pd
from gensim.test.utils import get_tmpfile
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

txt=pd.read_csv('./out/020train_w2v_temp_data.txt',sep='\t',header=None)
words=[]
for i in range(len(txt)):
    description=str(txt[0][i])
    description_split=description.split(" ")
    words.append(description_split)
#print words

path=get_tmpfile('word2vec.model')

model=word2vec.Word2Vec(min_count=1,size=200)

model.build_vocab(words)
model.train(words,total_examples=model.corpus_count,epochs=model.iter)
print(model)
try:
    print(model['HSP90AB2P'])
except:
    print("HSP90AB2P not in model")
'''
model.build_vocab([['HSP90AB2P']],update=True)
model.train([['HSP90AB2P']],total_examples=model.corpus_count,epochs=model.iter)
print ("last:")
print (model['HSP90AB2P'])
'''
model.save("./out/020Word2vec_Modifiable.model")
