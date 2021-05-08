#coding:utf-8
import gensim
import pandas as pd
context = pd.read_csv("./017not_in_w2v.txt",sep='\t',header=None)

words=[]
for i in range(len(context)):
    words.append(context[0][i])

ls=[]
ls.append(words)

model=gensim.models.word2vec.Word2Vec.load('./out/022Word2vec_model')

try:
    print("model['HSP90AB2P']")
    print(model['HSP90AB2P'])
except:
    print ("can't find 'HSP90AB2P")


model.build_vocab(ls,update=True)

model.train(ls,total_examples=model.corpus_count,epochs=model.iter)
print(model)
try:
    print("model['HSP90AB2P']")
    print(model['HSP90AB2P'])
except:
    print ("can't find 'HSP90AB2P")

model.save('./out/023Word2vec_model_modified_by_wiki_can_update_of_during_subsequent_training')
