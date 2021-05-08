# coding=utf-8
from sklearn import metrics
from keras.layers import Reshape
from keras.callbacks import EarlyStopping
from keras import Input, Model
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Concatenate
import pandas as pd
import numpy as np
import gensim
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from sklearn.model_selection import KFold

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)

class TextCNN(object):
    def get_model(self):
        input = Input(shape=[19200, ])
        input_reshape = Reshape((96, 200))(input)
        convs = []
        for kernel_size in [3, 5, 7, 9, 11, 13]:
            c = Conv1D(128, kernel_size, activation='relu')(input_reshape)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)
        class_num = 1
        output = Dense(units=class_num, activation="sigmoid")(x)
        model = Model(inputs=input, outputs=output)
        model.summary()
        return model


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
    word2vec_path_train = './out/03721Word2vec_word2vec_format_model'
    model_train = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    #model = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)  ######################################################

    max_words_set=set()

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
            for word in words:
                max_words_set.add(word)
            if len(words) > length:
                length = len(words)
    print("length" + str(length))
    XX = []
    ###############
    for i in range(len(sentX)):
            sent = sentX[i]
            sent_vec=get_sent_vec(200, length, sent, model, model_train)
            XX.append([sent_vec])
            i += 1

    XX = np.concatenate(XX)
    y = np.load('./out/035_36_sample.npy')
    return XX, y
if __name__ == '__main__':
    X, y = load_file()
    #早停法 early_stopping
    early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    for num in range(3):
        cv = StratifiedKFold(n_splits=10)
        #floder = KFold(n_splits=10, random_state=5 * num, shuffle=True)
        #for train_loc, test_loc in floder.split(X, y):
        for _, (train_loc, test_loc) in enumerate(cv.split(X, y)):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(X[train_loc])#X[train_loc]  # scaler.fit_transform(X[train])
            x_test = scaler.transform(X[test_loc])#X[test_loc]  # scaler.transform(X[test])
            y_train = y[train_loc]
            y_test = y[test_loc]

            batch_size = 10
            epochs = 100

            model = TextCNN().get_model()
            model.compile('adam', 'binary_crossentropy',metrics=['accuracy'])
            model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=0)
            #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping],validation_data=(X_test, y_test))
            result = model.evaluate(x_test,y_test,verbose=0)

            model_predict = np.concatenate(np.array(model.predict(x_test)))
            temp=[]
            for pred in model_predict:
                if pred >= 0.5:
                    temp.append(1.0)
                else:
                    temp.append(0.0)
            model_predict = np.array(temp)

            precision = metrics.precision_score(y_true=y_test, y_pred=model_predict)
            recall = metrics.recall_score(y_true=y_test, y_pred=model_predict)
            f1_score = metrics.f1_score(y_true=y_test, y_pred=model_predict)
            f = open("./out/035textCNN.txt", "a+")
            f.write(str(result[1]) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(f1_score) + "\t\n")
            f.close()