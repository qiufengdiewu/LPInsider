# coding=utf-8
import numpy as np
import pandas as pd
import gensim
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense,LSTM
from keras.models import Sequential
from sklearn import metrics
from sklearn.model_selection import KFold

def nomalization(X):
    return preprocessing.scale(X, axis=0)

# 计算词向量
def get_sent_vec(size, npLength, sent, model, model_train):
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
    vec_zero = np.zeros(size).reshape(1, size)
    count = 0
    vec = []
    for word in words:
        try:
            vec.append(model[word].reshape(1,size))
            count += 1
        except:
            try:
                vec.append(model_train[word].reshape(1,size))
                count += 1
            except:
                continue

    for i in range(count,npLength):
        vec.append(vec_zero)
    vec = np.concatenate(vec)
    vec = np.concatenate(vec)
    return nomalization(vec)

def load_file():
    # 训练模型
    X = pd.read_csv("./out/035_36_sample.txt", sep='\t', header=None,encoding='ISO-8859-1')  ############################
    # 导入模型
    word2vec_path = "E:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
    word2vec_path_train = './out/023Word2vec_model_modified_by_wiki_can_update_of_during_subsequent_training'
    model_train = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    #model = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)  ######################################################

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
                length = len(words)  #########小样本数据集的单词最大长度是97
    print("length" + str(length))

    XX = []
    ###############
    for i in range(len(sentX)):
        sent = sentX[i]
        XX.append([get_sent_vec(200, length, sent, model, model_train)])
        i += 1

    XX = np.concatenate(XX)
    y = np.load('./out/035_36_sample.npy')
    return XX, y

def trans(y):
    y1 = [1, 0]
    y0 = [0, 1]
    y_ = []
    for item in y:
        if item == 1:
            y_.append(y1)
        else:
            y_.append(y0)
    return np.array(y_)

if __name__ == '__main__':
    X, y = load_file()
    length = 96
    vector = 200
    for num in range(3):
        #cv = StratifiedKFold(n_splits=10)
        floder = KFold(n_splits=10, random_state=5 * num, shuffle=True)
        for train_loc, test_loc in floder.split(X, y):
        #for i, (train, test) in enumerate(cv.split(X, y)):
            #scaler = StandardScaler()
            x_train = X[train_loc] #scaler.fit_transform(X[train])
            x_test = X[test_loc]#scaler.transform(X[test])
            y_train = trans(y[train_loc])
            y_test = trans(y[test_loc])
            x_train = x_train.reshape(x_train.shape[0], length, vector)
            x_test = x_test.reshape(x_test.shape[0], length, vector)
            units = 140
            model = Sequential()
            model.add(LSTM(units=units, input_shape=(length, vector)))
            model.add(Dense(2, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam',
                          metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=0)
            #model.summary()
            result = model.evaluate(x_test, y_test, batch_size=128, verbose=0)

            predict = model.predict(x_test)
            temp_predict = []
            for i in range(len(predict)):
                if predict[i][0] >= predict[i][1]:
                    temp_predict.append(1)
                else:
                    temp_predict.append(0)
            predict = np.array(temp_predict)

            temp_y_test=[]
            for i in range(len(y_test)):
                if y_test[i][0] == 1 and y_test[i][1] == 0:
                    temp_y_test.append(1)
                elif y_test[i][0] == 0 and y_test[i][1] == 1:
                    temp_y_test.append(0)
            y_test = np.array(temp_y_test)
            print(str(y_test)+"\n"+str(predict))
            precision = metrics.precision_score(y_test, predict)
            recall = metrics.recall_score(y_test, predict)
            f1_score = metrics.f1_score(y_test, predict)
            f = open("./out/036LSTM.txt", "a+", encoding="utf-8")
            f.write(str(result[1])+"\t"+str(precision)+"\t"+str(recall)+"\t"+str(f1_score)+"\t\n")
            f.close()
