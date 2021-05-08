#coding:utf-8

import numpy as np
from gensim.models import word2vec
from sklearn.externals import joblib
import jieba

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


#对单个句子进行情感判断
def svm_predict(sent):
    model = word2vec.Word2Vec.load('./out/008train_model.model')
    sent_cut = jieba.lcut(sent)
    sent_cut_vec = get_sent_vec(300,sent_cut,model)
    clf = joblib.load('./out/010svm_model.pkl')
    result = clf.predict(sent_cut_vec)

    if int(result[0] == 1):
        print(sent,'pos')
    else:
        print(sent,'neg')


#情感正式开始预测
#sent = 'this is test'
sent="Comparative experiments on learning information extractors for proteins and their interactions"
svm_predict(sent)