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
import joblib
# 计算词向量
# 包括计算对应的位置特征
def get_sent_vec(size, npLength, sent, model, model_train):
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

    vec = nomalization(vec)

    return vec


# 训练模型

X = pd.read_csv("./out/007X_with_entity_and_stanford_parser_preprocess.txt", sep='\t', header=None,
                encoding='ISO-8859-1')  #######
_025POS_transform_to_unite=pd.read_csv("./in/025POS_transform_to_unite_preprocess.txt",sep="\t",header=None,encoding="utf-8")
y = np.load('./out/007X_with_entity_and_stanford_parser_preprocess.npy')  ####

f_svm = open("./out/results/010svm_10_preprocess_w2v_stanfordparser.txt", 'w')  ###################################################
f_LogisticR = open("./out/results/010LogisticR_10_preprocess_w2v_stanfordparser.txt", 'w')  ###################################################
f_RandomF = open('./out/results/010RandomF_10_preprocess_w2v_stanfordparser.txt', 'w')  ###################################################
f_xgboost = open('./out/results/010xgboost_10_preprocess_w2v_stanfordparser.txt', 'w')  ###################################################)
f_lightGBM = open('./out/results/010lightGBM_10_preprocess_w2v_stanfordparser.txt', 'w')  ###################################################)

def train(X, y, count):
    # 导入模型
    word2vec_path = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
    word2vec_path_train = "./out/03721Word2vec_word2vec_format_model"
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    model_train = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)
    #model = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)#测试的时候使用
    for c in range(10):###################
        sentX = []
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
                    length = len(words)#########小样本数据集的单词最大长度是97
        print("length"+str(length))
        XX = []
        for i in range(len(sentX)):
            sent = sentX[i]
            sent_vec=get_sent_vec(200, length, sent, model, model_train)
            XX.append([sent_vec])
        XX = np.concatenate(XX)
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
            ###################
            #joblib.dump(LGBM, './out/010LGBM_model.pkl')



            f_lightGBM.write(str(accuracy_LGBM) + '\t')
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

def nomalization(X):
    return preprocessing.scale(X, axis=0)

count = 0
train(X, y, count)

f_svm.close()
f_LogisticR.close()
f_RandomF.close()
f_xgboost.close()
f_lightGBM.close()