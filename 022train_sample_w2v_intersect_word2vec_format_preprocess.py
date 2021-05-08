#coding:utf-8
import gensim
model=gensim.models.word2vec.Word2Vec.load("./out/020Word2vec_Modifiable_preprocess.model")
print(model)
print(model['the'])

model.intersect_word2vec_format("./out/021Word2vec_word2vec_format_model",binary=True,lockf=1.0)
print(model)
print(model['the'])
model.save('./out/022Word2vec_model_can_update_of_during_subsequent_training_preprocess.model')
"""
model=gensim.models.word2vec.Word2Vec.load('./out/022Word2vec_model')
print ("------------------------")
print (model['the'])
"""
