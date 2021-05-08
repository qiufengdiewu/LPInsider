# coding=utf-8
import gensim
import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

txt = pd.read_csv('./out/020train_w2v_temp_data.txt',sep='\t',header=None)
model = gensim.models.word2vec.Word2Vec.load("./out/020Word2vec_model")
model.wv.save_word2vec_format("./out/021Word2vec_word2vec_format_model",binary=True)

#read
model = gensim.models.KeyedVectors.load_word2vec_format("./out/021Word2vec_word2vec_format_model",binary=True)

word2vec_wiki = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
model_wiki = gensim.models.KeyedVectors.load_word2vec_format(word2vec_wiki, binary=True)


for i in range(len(txt)):
    description = str(txt[0][i])
    descriptionD = description.split(" ")
    for word in descriptionD:
        word_vector = []
        word_vector.append(str(word))
        model_vector = []
        try:
            model_vector.append(model_wiki[str(word)])
            model.add(word_vector, model_vector, replace=True)
            print(model_wiki[str(word)])
        except:
            print('the word'+str(word)+'does not in vocabulary')


model.wv.save_word2vec_format("./out/021Word2vec_word2vec_format_model",binary=True)
print("success")
