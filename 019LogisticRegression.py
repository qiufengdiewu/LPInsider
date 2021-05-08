#coding:utf-8
import numpy as np
import gensim
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#计算词向量
def get_sent_vec(size,sent,model,model_train):
    vec = np.zeros(size).reshape(1,size)
    count = 0
    sent=str(sent).replace(',',' ')
    sent=sent.replace('(',' ')
    sent=sent.replace(')',' ')
    sent=sent.replace("'",' ')
    sent=sent.replace('.',' ')
    sent=sent.replace(':',' ')
    sent=sent.replace(']',' ')
    sent=sent.replace('[', ' ')
    sent=sent.replace('/',' ')
    words=sent.split(" ")
    for word in words:
    #for word in sent:
        try:
            vec += model[word].reshape(1,size)
            count += 1
        except:
            try:
                vec += model_train[word].reshape(1,size)
                count += 1
            except:
                continue

    if count != 0:
        vec /= count
    return vec


#训练模型
#X = np.load('X.npy')
import pandas as pd
X=pd.read_csv("./out/007X.txt",sep='\t',header=None)
y = np.load('./out/007y.npy')

# 导入模型
word2vec_path = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
word2vec_path_train='./out/023Word2vec_model_modified_by_wiki'
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
model_train=gensim.models.word2vec.Word2Vec.load(word2vec_path_train)

def LogisticRegression_train(X, y):
    score = []
    for c in range(5):
        clf = LogisticRegression(C=100, max_iter=200)
        sentX = []
        for i in range(len(X)):
            sentX.append(X[0][i])

        XX=np.concatenate([get_sent_vec(200,sent,model,model_train) for sent in sentX])

        train_vec, test_vec, y_train, y_test = train_test_split(XX, y, test_size=0.2, random_state=5 + c * 11)

        train_vec = nomalization(train_vec)
        test_vec = nomalization(test_vec)

        clf.fit(train_vec, y_train)
        # 持久化保存模型
        joblib.dump(clf, './out/019LogisticRegression_train_model.pkl', compress=3)


        #print(clf.score(y_train,y_test))
        acc = clf.score(test_vec, y_test)
        score.append(acc)
    return np.mean(score)


def nomalization(X):
    return preprocessing.scale(X, axis=0)


#score = LogisticRegression_train(X, y)
#print "LogisticRegression_train"+str(score)


def RandomForestClassifier_train(X, y):
    score = []
    for c in range(5):
        forest = RandomForestClassifier(criterion='entropy', n_estimators=1000)
        sentX = []
        for i in range(len(X)):
            sentX.append(X[0][i])

        XX=np.concatenate([get_sent_vec(200,sent,model,model_train) for sent in sentX])

        train_vec, test_vec, y_train, y_test = train_test_split(XX, y, test_size=0.2, random_state=5 + c * 11)

        train_vec = nomalization(train_vec)
        test_vec = nomalization(test_vec)

        forest.fit(train_vec, y_train)
        # 持久化保存模型
        joblib.dump(forest, './out/019RandomForestClassifier_train_model.pkl', compress=3)
        score = forest.predict_proba(test_vec)
        prob = forest.predict(test_vec)
        acc=forest.score(test_vec, y_test)
        #score.append(acc)
        print(acc)
    #return np.mean(score)


score = RandomForestClassifier_train(X, y)
print ("RandomForestClassifier_train"+str(score))
