#coding:utf-8

import numpy as np
from gensim.models import word2vec
from sklearn.externals import joblib
import jieba
import gensim

#计算词向量
def get_sent_vec(size,sent,model):
    vec = np.zeros(size).reshape(1,size)
    count = 0
    for word in sent:
        try:
            vec += model[word].reshape(1,size)
            count += 1
        except:
            continue
    if count != 0:
        vec /= count
    return vec

#导入模型
word2vec_path="I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
model= gensim.models.KeyedVectors.load_word2vec_format(word2vec_path,binary=True)
#对单个句子进行情感判断
def svm_predict(sent,model):
    sent_cut = jieba.lcut(sent)
    sent_cut_vec = get_sent_vec(200,sent_cut,model)
    #clf = joblib.load('./out/010svm_model_wiki.pkl')
    clf = joblib.load('./out/010svm_model.pkl')
    result = clf.predict(sent_cut_vec)
    if int(result[0] == 1):
        print(sent,'pos')
    else:
        print(sent,'neg')

#情感正式开始预测

sent = 'this is test'
#sent="Comparative experiments on learning information extractors for proteins and their interactions"
svm_predict(sent,model)