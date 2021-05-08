#coding:utf-8
import gensim
model = gensim.models.word2vec.Word2Vec.load("./out/020Word2vec_Modifiable.model")
print(model)
print(model['the'])
print("++++++++++++++++++++++++")
#model_modified_by_wiki=gensim.models.KeyedVectors.load_word2vec_format("./out/021Word2vec_word2vec_format_model",binary=True)

model.intersect_word2vec_format("./out/021Word2vec_word2vec_format_model",binary=True,lockf=1.0)

print(model['the'])
model.save('./out/022Word2vec_model_can_update_of_during_subsequent_training.model')
'''
model=gensim.models.word2vec.Word2Vec.load('./out/022Word2vec_model')
print ("------------------------")
print (model['the'])
'''