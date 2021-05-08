# coding:utf-8
import pandas as pd
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

# 计算词向量
# 包括计算对应的位置特征
def get_sent_vec(size, npLength, sent, model, model_train, lncRNA,protein,sent_POS,model_POS):
    length_POS=npLength
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
    lncRNA_location = words.index(lncRNA)
    protein_location = words.index(protein)
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
        if words_POS!="":
            try:
                vec_word_POS = model_POS[word_POS].reshape(1, size)
                vec_POS = np.append(vec_POS, vec_word_POS)
                length_POS -= 1
            except:
                pass

    while length_POS>=0:
        vec_POS=np.append(vec, np.zeros(size).reshape(1, size))
        length_POS-=1

    #####################
    vec=nomalization(vec)
    lncRNA_matrix=nomalization(lncRNA_matrix)
    protein_matrix=nomalization(protein_matrix)
    vec_POS=nomalization(vec_POS)
    vec=np.concatenate((vec,lncRNA_matrix,protein_matrix,vec_POS),axis=0)
    return vec

# 训练模型

X = pd.read_csv("./out/007X_with_entity_and_stanford_parser.txt", sep='\t', header=None,
                encoding='ISO-8859-1')  ############################
y = np.load('./out/007X_with_entity_and_stanford_parser.npy')  ###################################################
_025POS_transform_to_unite=pd.read_csv("./in/025POS_transform_to_unite.txt",sep="\t",header=None,encoding="utf-8")

f_svm = open("./out/010svm_8.txt", 'w')  ###################################################
f_LogisticR = open("./out/010LogisticR_8.txt", 'w')  ###################################################
f_RandomF = open('./out/010RandomF_8.txt', 'w')  ###################################################
f_xgboost = open('./out/010xgboost_8.txt', 'w')  ###################################################)
f_lightGBM = open('./out/010lightGBM_8.txt', 'w')  ###################################################)

def train(X, y, count):
    # 导入模型
    word2vec_path = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
    #word2vec_path_train = './out/023Word2vec_model_modified_by_wiki'
    word2vec_path_train='./out/023Word2vec_model_modified_by_wiki_can_update_of_during_subsequent_training'
    # model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    model_train = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)
    model = gensim.models.word2vec.Word2Vec.load(
        word2vec_path_train)  ######################################################
    model_POS=gensim.models.word2vec.Word2Vec.load("./out/0263abstract2POS_united2_w2v.model")#####("./out/024POS_of_words.model")

    for c in range(1):###################
        sentX = []
        sentX_POS=[]
        length = 0
        for i in range(0, len(X), 1):
            sentX.append(X[2][i])
            for sent in sentX:
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
                if len(words) > length:
                    length = len(words)
        for i in range(len(_025POS_transform_to_unite)):
            sentX_POS.append(_025POS_transform_to_unite[2][i])

        ###############
        XX = []
        for i in range(len(sentX)):
            sent=sentX[i]
            sent_POS=sentX_POS[i]
            print()
            XX.append([get_sent_vec(200, length, sent, model, model_train,X[0][i],X[1][i],sent_POS,model_POS)])
            #XX.append([get_sent_vec(200, length, sent, model, model_train,X[0][i],X[1][i],length_POS,sent_POS,POS_classified0,POS_matrix,length_classified)])
            i+=1

        XX=np.concatenate(XX)

        ####################################
        floder = KFold(n_splits=10, random_state=5 * c, shuffle=True)
        for train_loc, test_loc in floder.split(XX, y):
            train_vec = XX[train_loc]

            y_train = y[train_loc]
            test_vec = XX[test_loc]
            y_test = y[test_loc]

            print("lightGBM")
            # lightGBM############################################
            LGBM = lightgbm.LGBMClassifier()
            LGBM.fit(train_vec, y_train)
            accuracy_LGBM = LGBM.score(test_vec, y_test)
            predict_LGBM = LGBM.predict(test_vec)
            precision_LGBM = metrics.precision_score(y_test, predict_LGBM)
            recall_LGBM = metrics.recall_score(y_test, predict_LGBM)
            f1_LGBM = metrics.f1_score(y_test, predict_LGBM)

            f_lightGBM.write(str(accuracy_LGBM) + '\t')
            print(accuracy_LGBM)
            f_lightGBM.write(str(precision_LGBM) + '\t' + str(recall_LGBM) + '\t' + str(f1_LGBM) + '\n')

            # xgboost###############################################

            reg = xgboost.XGBClassifier(silent=1)
            reg.fit(train_vec, y_train)

            accuracy_XGB = reg.score(test_vec, y_test)
            predict_XGB = reg.predict(test_vec)

            precision_XGB = metrics.precision_score(y_test, predict_XGB)
            recall_XGB = metrics.recall_score(y_test, predict_XGB)
            f1_XGB = metrics.f1_score(y_test, predict_XGB)

            f_xgboost.write(str(accuracy_XGB) + '\t')
            f_xgboost.write(str(precision_XGB) + '\t' + str(recall_XGB) + '\t' + str(f1_XGB) + '\n')

            # svm###################################################
            clf_svm = SVC(kernel='rbf', verbose=True, C=10)
            clf_svm.fit(train_vec, y_train)
            accuracy_SVM = clf_svm.score(test_vec, y_test)
            predict = clf_svm.predict(test_vec)

            precision_SVM = metrics.precision_score(y_test, predict)
            recall_SVM = metrics.recall_score(y_test, predict)
            f1_SVM = metrics.f1_score(y_test, predict)
            f_svm.write(str(accuracy_SVM) + '\t')
            f_svm.write(str(precision_SVM) + '\t' + str(recall_SVM) + '\t' + str(f1_SVM) + '\n')

            # 逻辑回归##################################################
            clf_LogR = LogisticRegression(C=100, max_iter=200)
            clf_LogR.fit(train_vec, y_train)
            accuracy_LogR = clf_LogR.score(test_vec, y_test)
            predict_logR = clf_LogR.predict(test_vec)

            precision_logR = metrics.precision_score(y_test, predict_logR)
            recall_logR = metrics.recall_score(y_test, predict_logR)
            f1_logR = metrics.f1_score(y_test, predict_logR)

            f_LogisticR.write(str(accuracy_LogR) + '\t')
            f_LogisticR.write(str(precision_logR) + '\t' + str(recall_logR) + '\t' + str(f1_logR) + '\n')

            # RandomForestClassifier ##################################################
            forest = RandomForestClassifier(criterion='entropy', n_estimators=1000)
            forest.fit(train_vec, y_train)
            acc_RF = forest.score(test_vec, y_test)
            predict_RF = forest.predict(test_vec)

            precision_RF = metrics.precision_score(y_test, predict_RF)
            recall_RF = metrics.recall_score(y_test, predict_RF)
            f1_RF = metrics.f1_score(y_test, predict_RF)

            f_RandomF.write(str(acc_RF) + '\t')
            f_RandomF.write(str(precision_RF) + '\t' + str(recall_RF) + '\t' + str(f1_RF) + '\n')
            count += 1
            print("#################success:" + str(int(c) + 1) + ' ' + str(count))
    # return np.mean(score, axis=0)


def nomalization(X):
    return preprocessing.scale(X, axis=0)


count = 0
train(X, y, count)

f_svm.close()
f_LogisticR.close()
f_RandomF.close()
f_xgboost.close()
f_lightGBM.close()

