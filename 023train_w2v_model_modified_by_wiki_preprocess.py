# coding=utf-8
import gensim
import pandas as pd
context = pd.read_csv("./out/017not_in_w2v_preprocess.txt",sep='\t',header=None)
words = []
for i in range(len(context)):
    words.append(context[0][i])

ls = []
ls.append(words)

model = gensim.models.word2vec.Word2Vec.load('./out/022Word2vec_model')
print(model)
print(model['the'])
model.build_vocab(ls, update=True)
model.train(ls, total_examples=model.corpus_count, epochs=model.iter)
model.save('./out/023Word2vec_model_modified_by_wiki_preprocess')
print(model)
print(model['the'])
print("success")