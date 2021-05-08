# coding=utf-8

import stanfordcorenlp
path_nlp='I:/stanford_parser/stanford-corenlp-full-2018-10-05'
nlp=stanfordcorenlp.StanfordCoreNLP(path_nlp)
import pandas as pd
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import re
import string
import joblib

import numpy as np
import gensim
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import xgboost
import lightgbm
from sklearn.metrics.pairwise import pairwise_distances
path = "I:/Code/Django/project/junicer"

def sentence_vec(sentence,lncRNA,protein):

    # 导入模型
    word2vec_path = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
    # word2vec_path_train = './out/023Word2vec_model_modified_by_wiki'
    word2vec_path_train = path + '/out/023Word2vec_model_modified_by_wiki_can_update_of_during_subsequent_training'
    # model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    model_train = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)
    model = gensim.models.word2vec.Word2Vec.load(
        word2vec_path_train)  ######################################################
    model_POS = gensim.models.word2vec.Word2Vec.load(path + "/out/024POS_of_words.model")
    length = 97
    length_POS = 97
    lncRNAs=[]
    proteins=[]

    ############计算词性矩阵，例如NN：[0,1,0,0,0,0,0,0,0,0,0]
    POS_classified = pd.read_csv(path+"/in/POS_classified.txt", sep='\t', header=None)
    length_classified = len(POS_classified)
    POS_classified0 = POS_classified[0]
    POS_matrix = np.zeros((length_classified, length_classified))
    for i in range(length_classified):
        POS_matrix[i][i] = 1

    ###############
    XX=[]

    sent = sentence

    sent_POS = sentX_POS(sentence)##############计算词性

    XX.append([get_sent_vec(200, length, sent, model, model_train, lncRNA, protein, length_POS, sent_POS,
                            POS_classified0, POS_matrix, length_classified)])

    XX = np.concatenate(XX)

    LGBM_model = joblib.load(path + "/out/010LGBM_model.pkl")
    pred = LGBM_model.predict(XX)
    print(float(pred[0]))
    print(type(float(pred[0])))
    print()


def sentX_POS(sentence):
    description = str(sentence)
    description = description.replace('(', ' ')
    description = description.replace(',', ' ')
    description = description.replace(')', ' ')
    description = description.replace('.', ' ')
    description = description.replace("'", ' ')
    description = description.replace(':', ' ')
    description = description.replace('[', ' ')
    description = description.replace(']', ' ')
    description = description.replace('/', ' ')
    sentence = description
    sentence_pos=nlp.pos_tag(sentence)
    POSs=""
    for i in range(len(sentence_pos)):
        POSs+=str(sentence_pos[i][1])+" "

    sentence_pos=POSs.split(" ")

    POS_classified = pd.read_csv(path+"./in/POS_classified.txt", sep='\t', header=None)
    length_classified = len(POS_classified)
    POS_classified0 = POS_classified[0]
    POS_classified1 = POS_classified[1]
    for i in range(length_classified - 1):
        POS_classified1[i] = POS_classified1[i].split(" ")

    POS_unite=""
    for k in range(len(sentence_pos)):
        word=sentence_pos[k]
        temp_unite=""
        for j in range(length_classified-1):
            flag=0
            for m in range(len(POS_classified1[j])):
                if word == POS_classified1[j][m]:
                    temp_unite=POS_classified0[j]
                    flag=1
                    break
                else:
                    temp_unite=POS_classified0[length_classified-1]
            if flag==1:
                break
        POS_unite+=(temp_unite+" ")
    return POS_unite


# 计算词向量
# 包括计算对应的位置特征
def get_sent_vec(size, npLength, sent, model, model_train, lncRNA,protein,length_POS,sent_POS,POS_classified0,POS_matrix,length_classified):
    vec = []
    sent = str(sent).replace(',', ' ')
    sent = sent.replace('(', ' ')
    sent = sent.replace(')', ' ')
    sent = sent.replace("'", ' ')
    sent = sent.replace('.', ' ')
    sent = sent.replace(':', ' ')
    sent = sent.replace(']', ' ')
    sent = sent.replace('[', ' ')
    sent = sent.replace('/', ' ')
    words = sent.split(" ")
    for word in words:
        try:
            vec_word = model[word].reshape(1, size)
            vec = np.append(vec, vec_word)
            npLength -= 1
        except:
            try:
                vec_word = model_train[word].reshape(1, size)
                vec = np.append(vec, vec_word)
                npLength -= 1
            except:
                continue
    while npLength >= 0:
        vec = np.append(vec, np.zeros(size).reshape(1, size))
        npLength -= 1

    # 计算位置特征
    matrix = np.zeros((1, 6))
    lncRNA_matrix = matrix[0]
    protein_matrix = matrix[0]
    if lncRNA == "5'aHIF1alpha":
        words[words.index('aHIF1alpha')] = "5'aHIF1alpha"
    try:
        lncRNA_location = words.index(lncRNA)
    except:
        lncRNA_location = -1
    try:
        protein_location = words.index(protein)
    except:
        protein_location = -1
    try:
        lncRNA_w2v = model_train[lncRNA]
        protein_w2v = model_train[protein]

        count = 0
        # 计算lncRNA的距离矩阵
        for i in range(lncRNA_location - 1, -1, -1):
            try:
                word_w2v = model_train[words[i]]
                lncRNA_matrix[2 - count] = pairwise_distances([lncRNA_w2v, word_w2v])[0][1]
                count += 1
                if count >= 3:
                    break
            except:
                pass
        count = 0
        for i in range(lncRNA_location + 1, len(words)):
            try:
                word_w2v = model_train[words[i]]
                lncRNA_matrix[3 + count] = pairwise_distances([lncRNA_w2v, word_w2v])[0][1]
                count += 1
                if count >= 3:
                    break
            except:
                pass
        # 计算protein的距离矩阵
        # 这里可以写成一个函数，减少行数，but我没改。emmm
        count = 0
        for i in range(protein_location - 1, -1, -1):
            try:
                word_w2v = model_train[words[i]]
                protein_matrix[2 - count] = pairwise_distances([protein_w2v, word_w2v])[0][1]
                count += 1
                if count >= 3:
                    break
            except:
                pass

        count = 0
        for i in range(protein_location + 1, len(words)):
            try:
                word_w2v = model_train[words[i]]
                protein_matrix[3 + count] = pairwise_distances([protein_w2v, word_w2v])[0][1]
                count += 1
                if count >= 3:
                    break
            except:
                pass

    except:
        print("first try::::::::::::: except")
        pass


    ######计算词性特征
    vec_POS=[]
    words_POS=str(sent_POS).split(" ")
    for word_POS in words_POS:
        #POS_classified0,POS_matrix,length_classified
        for i in range(length_classified):
            if str(word_POS)==str(POS_classified0[i]):
                vec_POS=np.append(vec_POS,POS_matrix[i])
                length_POS-=1
                break

    while length_POS>=0:
        vec_POS=np.append(vec_POS,np.zeros(length_classified).reshape(1,length_classified))
        length_POS-=1

    #####################
    vec=nomalization(vec)
    lncRNA_matrix=nomalization(lncRNA_matrix)
    protein_matrix=nomalization(protein_matrix)
    vec_POS=nomalization(vec_POS)
    vec=np.concatenate((vec,lncRNA_matrix,protein_matrix,vec_POS),axis=0)
    return vec



def nomalization(X):
    return preprocessing.scale(X, axis=0)




if __name__ == '__main__':
    sentence="Among them, H19 is recently discovered as a class of lncRNAs which is related to fibrotic disease and inflammation P53"
    lncRNA="H19"
    protein="p53"
    sentence_vec(sentence,lncRNA,protein)


