# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Model
from keras.layers import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras import activations
from keras import backend as K
from keras.engine.topology import Layer
import pandas as pd
import gensim
from sklearn import preprocessing
from sklearn import metrics
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

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x


#define our own softmax function instead of K.softmax
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


#A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        #keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def trans(y):
    y1 = [1, 0]
    y0 = [0, 1]
    y_ = []

    for item in y:
        if item == 1:
            y_.append(y1)
        else:
            y_.append(y0)
    # print(y_train.__len__())
    return np.array(y_)

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
    # word2vec_path_train = './out/023Word2vec_model_modified_by_wiki'
    word2vec_path_train = './out/023Word2vec_model_modified_by_wiki_can_update_of_during_subsequent_training'
    model_train = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)
    #model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    model = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)

    for c in range(1):  ###################
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
                    length = len(words)
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
if __name__ == '__main__':
    X, y = load_file()
    length = int(len(X[0]))
    cv = StratifiedKFold(n_splits=10)
    for num in range(3):
        floder = KFold(n_splits=10, random_state=5 * num, shuffle=True)
        for train_loc, test_loc in floder.split(X, y):
            x_train = X[train_loc]  # scaler.fit_transform(X[train])
            x_test = X[test_loc]  # scaler.transform(X[test])
            y_train = trans(y[train_loc])
            y_test = trans(y[test_loc])
            """    
            for _, (train, test) in enumerate(cv.split(X, y)):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train])
                X_test = scaler.transform(X[test])
                y_train = trans(y[train])
                y_test = trans(y[test])
                X_train = X_train.reshape(X_train.shape[0], 96, 200, 1)
                X_test = X_test.reshape(X_test.shape[0], 96, 200, 1)
            """
            # 搭建CNN+Capsule分类模型
            input_image = Input(shape=(None, None, 1))
            cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
            cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
            cnn = MaxPooling2D((2, 2))(cnn)
            cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
            cnn = Conv2D(128, (3, 3), activation='relu')(cnn)

            cnn = Reshape((-1, 128))(cnn)
            capsule = Capsule(2, 16, 3, True)(cnn)
            output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(2,))(capsule)
            model = Model(inputs=input_image, outputs=output)

            model.compile(loss=lambda y_true, y_pred: y_true * K.relu(0.9 - y_pred) ** 2 + 0.25 * (1 - y_true) * K.relu(
                y_pred - 0.1) ** 2, optimizer='adam',
                          metrics=['accuracy'])

            model.summary()
            model.fit(x_train, y_train, epochs=100,batch_size=10, verbose=0)#model.fit(X_train, y_train,batch_size=43,epochs=100,verbose=0)

            results = model.evaluate(x_test, y_test)
            predict = model.predict(x_test)
            temp_predict = []
            for i in range(len(predict)):
                if predict[i][0] >= predict[i][1]:
                    temp_predict.append(1)
                else:
                    temp_predict.append(0)
            predict = np.array(temp_predict)

            temp_y_test = []
            for i in range(len(y_test)):
                if y_test[i][0] == 1 and y_test[i][1] == 0:
                    temp_y_test.append(1)
                elif y_test[i][0] == 0 and y_test[i][1] == 1:
                    temp_y_test.append(0)
            y_test = np.array(temp_y_test)
            print(str(y_test) + "\n" + str(predict))
            precision = metrics.precision_score(y_test, predict)
            recall = metrics.recall_score(y_test, predict)
            f1_score = metrics.f1_score(y_test, predict)
            f = open("./out/036capsule_network1.txt", "a+", encoding="utf-8")
            f.write(str(results[1])+"\t"+str(precision)+"\t"+str(recall)+"\t"+str(f1_score)+"\t\n")
            f.close()
