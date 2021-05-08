# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Model
from keras.layers import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from keras import activations
from keras import backend as K
from keras.engine.topology import Layer
import pandas as pd
import gensim
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
import tensorflow as tf

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

    vec = np.zeros(size).reshape(1, size)
    count = 0
    for word in words:
        try:
            vec+=model[word].reshape(1,size)
            count+=1
        except:
            continue
    if count!=0:
        vec/=count

    # 计算位置特征
    matrix = np.zeros((1, 6))
    lncRNA_matrix = matrix[0]
    protein_matrix = matrix[0]
    if lncRNA == "5'aHIF1alpha":
        words[words.index('aHIF1alpha')] = "5'aHIF1alpha"
    try:
        lncRNA_location = words.index(lncRNA)
    except:
        lncRNA_location=-1

    try:
        protein_location = words.index(protein)
    except:
        protein_location=-1

    try:
        try:
            lncRNA_w2v = model[lncRNA]
            protein_w2v = model[protein]
        except:
            lncRNA_w2v = model_train[lncRNA]
            protein_w2v = model_train[protein]

        count = 0
        # 计算lncRNA的距离矩阵
        for i in range(lncRNA_location - 1, -1, -1):
            try:
                word_w2v = model[words[i]]
                lncRNA_matrix[2 - count] = pairwise_distances([lncRNA_w2v, word_w2v])[0][1]
                count += 1
                if count >= 3:
                    break
            except:
                pass
        count = 0
        for i in range(lncRNA_location + 1, len(words)):
            try:
                word_w2v = model[words[i]]
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
                word_w2v = model[words[i]]
                protein_matrix[2 - count] = pairwise_distances([protein_w2v, word_w2v])[0][1]
                count += 1
                if count >= 3:
                    break
            except:
                pass

        count = 0
        for i in range(protein_location + 1, len(words)):
            try:
                word_w2v = model[words[i]]
                protein_matrix[3 + count] = pairwise_distances([protein_w2v, word_w2v])[0][1]
                count += 1
                if count >= 3:
                    break
            except:
                pass

    except:
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
    vec=vec[0]
    vec=nomalization(vec)
    lncRNA_matrix=nomalization(lncRNA_matrix)
    protein_matrix=nomalization(protein_matrix)
    vec_POS=nomalization(vec_POS)

    vec=np.concatenate((vec,lncRNA_matrix,protein_matrix,vec_POS),axis=0)
    return nomalization(vec)


def load_file():
    # 训练模型
    X = pd.read_csv("./out/007X_with_entity_and_stanford_parser.txt", sep='\t', header=None,
                    encoding='ISO-8859-1')  ############################
    _025POS_transform_to_unite = pd.read_csv("./in/025POS_transform_to_unite.txt", sep="\t", header=None,
                                             encoding="utf-8")


    # 导入模型
    word2vec_path = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
    # word2vec_path_train = './out/023Word2vec_model_modified_by_wiki'
    word2vec_path_train = './out/023Word2vec_model_modified_by_wiki_can_update_of_during_subsequent_training'

    model_train = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)
    #model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    model = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)  ######################################################
    model_POS = gensim.models.word2vec.Word2Vec.load("./out/024POS_of_words.model")

    matrix = np.zeros((1, 6))

    for c in range(1):  ###################
        sentX = []
        sentX_POS = []
        length = 0
        length_POS = 0
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

        for i in range(len(_025POS_transform_to_unite)):
            sentX_POS.append(_025POS_transform_to_unite[2][i])
            for sent in sentX_POS:
                words = str(sent).split(" ")
                if len(words) > length_POS:
                    length_POS = len(words)

        print("length_POS" + str(length_POS))
        lncRNAs = []
        proteins = []
        for i in range(len(X)):
            lncRNAs.append(X[0][i])
            proteins.append(X[1][i])
        XX = []
        i = 0
        '''
        leng1=len(sentX)
        leng2=len(sentX_POS)
        #此时leng1和leng2相等，只需要用一个就OK了
        '''
        ############计算词性矩阵，例如NN：[0,1,0,0,0,0,0,0,0,0,0]
        POS_classified = pd.read_csv("./in/POS_classified.txt", sep='\t', header=None)
        length_classified = len(POS_classified)
        POS_classified0 = POS_classified[0]
        POS_matrix = np.zeros((length_classified, length_classified))
        for i in range(length_classified):
            POS_matrix[i][i] = 1

        ###############
        for i in range(len(sentX)):
            sent = sentX[i]
            sent_POS = sentX_POS[i]
            XX.append([get_sent_vec(200, length, sent, model, model_train, X[0][i], X[1][i], length_POS, sent_POS,
                                    POS_classified0, POS_matrix, length_classified)])
            i += 1

        XX = np.concatenate(XX)
    y = np.load('./out/007X_with_entity_and_stanford_parser.npy')
    return XX, y


X, y = load_file()
#X = X[:, :400]
print(X.shape)
length=int(len(X[0]))

#X=X[:,:400]

#X = np.load('./original_data/kmer.npy')[:, :400]###########
#y = np.load('./original_data/y.npy')##################

cv = StratifiedKFold(n_splits=10)
acc = []
for i, (train, test) in enumerate(cv.split(X, y)):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train])
    X_test = scaler.transform(X[test])

    y_train = trans(y[train])
    y_test = trans(y[test])
    #temp=X_train.shape[0]

    X_train = X_train.reshape(X_train.shape[0], 43, 30, 1)  ##############
    X_test = X_test.reshape(X_test.shape[0], 43, 30, 1)  ######################

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
        y_pred - 0.1) ** 2,
                  optimizer='adam',
                  metrics=['accuracy'])

    #model.summary()

    model.fit(X_train, y_train,epochs=200, verbose=0)#model.fit(X_train, y_train,batch_size=43,epochs=100,verbose=0)

    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)
    acc.append(score[1])

print(np.mean(acc))
print(acc)


