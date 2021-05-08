#conding=utf-8
import pandas as pd
from gensim.models import Word2Vec
model=Word2Vec(size=200,min_count=1)
POS=pd.read_csv("./out/0262abstract2POS_united.txt",sep="\t",header=None)
titles=[]
abstracts=[]
for i in range(len(POS)):
    title=str(POS[0][i]).split(" ")
    abstract=str(POS[1][i]).split(" ")
    temp_title=[]
    temp_abstract=[]
    for j in range(len(title)):
        if title[j]!="":
            temp_title.append(title[j])
    titles.append(temp_title)

    for j in range(len(abstract)):
        if abstract[j]!="":
            temp_abstract.append(abstract[j])
    abstracts.append(temp_abstract)

sentences=titles+abstracts
model.build_vocab(sentences)
model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
print(model)
model.save("./out/0263abstract2POS_united2_w2v.model")