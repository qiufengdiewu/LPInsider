# coding=utf-8
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

not_in_wiki_w2v=pd.read_csv("./out/017not_in_w2v_preprocess.txt",sep="\t",header=None)
words_all_not_in_wiki=[]
for i in range(len(not_in_wiki_w2v)):
    word = str(not_in_wiki_w2v[0][i])
    words_all_not_in_wiki.append(word)

temp_txt = open("./out/017in_w2v_preprocess.txt","r")
model = word2vec.Word2Vec(LineSentence(temp_txt),min_count=1,size=200)
model_temp = model

model.wv.save_word2vec_format("./out/03721Word2vec_word2vec_format_model",binary=True)
#read
model = gensim.models.KeyedVectors.load_word2vec_format("./out/03721Word2vec_word2vec_format_model",binary=True)
print("first")
print(model)
print(model["H19"])
words_all = []
in_w2v = pd.read_csv("./out/017in_w2v_preprocess.txt",sep="\t",header=None)
for i in range(len(in_w2v)):
    sentence = str(in_w2v[0][i])
    words=sentence.split(" ")
    for word in words:
        if len(word)>0:
            words_all.append(word)

word2vec_wiki = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
model_wiki = gensim.models.KeyedVectors.load_word2vec_format(word2vec_wiki, binary=True)
for word in words_all:
    word_vector=[]
    word_vector.append(str(word))
    model_vector = []
    #try:
    word_vector_wiki = model_wiki[str(word)]
    model_vector.append(word_vector_wiki)
    model.add(word_vector, model_vector, replace=True)
    #except:
        #print('the word ' + str(word) + ' does not in wikipedia-pubmed-and-PMC-w2v.bin')
print("wiki")
print(model)
print(model["H19"])
print(model_wiki["H19"])
model.wv.save_word2vec_format("./out/03721Word2vec_word2vec_format_model",binary=True)
model_temp.intersect_word2vec_format("./out/021Word2vec_word2vec_format_model",binary=True,lockf=1.0)
#model.intersect_word2vec_format("./out/021Word2vec_word2vec_format_model",binary=True,lockf=1.0)
print("final")
print(model_temp["H19"])
model_temp.save("./out/03721Word2vec_word2vec_format_model")
#read
model = gensim.models.word2vec.Word2Vec.load("./out/03721Word2vec_word2vec_format_model")
model.build_vocab(words_all_not_in_wiki,update=True)
model.train(words_all_not_in_wiki,total_examples=model.corpus_count,epochs=model.iter)

model.save("./out/03721Word2vec_word2vec_format_model")
print("success")
print(model["H19"])
