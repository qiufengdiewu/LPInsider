# coding=utf-8

from gensim.models import word2vec
import logging
import pandas as pd
from gensim.test.utils import get_tmpfile
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def preprocess(description):
    description = description.replace('(', ' ')
    description = description.replace(',', ' ')
    description = description.replace(')', ' ')
    description = description.replace('.', ' ')
    description = description.replace("'", ' ')
    description = description.replace(':', ' ')
    description = description.replace('[', ' ')
    description = description.replace(']', ' ')
    description = description.replace('/', ' ')
    return description

txt=pd.read_csv('./out/017in_w2v_preprocess.txt',sep='\t',header=None)
words = []
for i in range(len(txt)):
    description = str(txt[0][i])
    description = preprocess(description)
    description_split = description.split(" ")
    words.append(description_split)


path=get_tmpfile('word2vec.model')
model=word2vec.Word2Vec(min_count=1,size=200)
model.build_vocab(words)
model.train(words,total_examples=model.corpus_count,epochs=model.iter)

try:
    print(model['HSP90AB2P'])
except:
    print("HSP90AB2P not in model")

model.save("./out/020Word2vec_Modifiable_preprocess.model")
print(model)