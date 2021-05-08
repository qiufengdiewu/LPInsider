# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pandas as pd
import gensim
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

class BiLSTM():
    def __init__(self, lr, timesteps, num_input, batch_size, num_hidden, epochs, lamda, i):
        self.lr = lr
        self.timesteps = timesteps
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_classes = 2
        self.batch_size = batch_size
        self.epochs = epochs
        self.lamda = lamda
        self.i = i

    def weight(self, shape):
        out = tf.truncated_normal(shape, stddev=0.1)
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.lamda)(out))
        return tf.Variable(out)

    def bias(self, shape):
        return tf.Variable(tf.truncated_normal(stddev=0.1, shape=shape))

    def clf(self, X_train, y_train, X_test, y_test):
        tf.reset_default_graph()
        tf.set_random_seed(seed=1)
        self.x = tf.placeholder(tf.float32, [None, self.timesteps, self.num_input])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        x = self.x
        y = self.y
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        x_ = tf.unstack(x, self.timesteps, 1)

        # 定义lstm单元cell
        # 前向传播的单元cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=0.6)
        # 反向传播的单元cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=0.6)

        # lstm单元的输出
        #lstm_tw_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_, dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x_,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output

        logits = tf.matmul(outputs[-1], self.weight([2 * self.num_hidden, self.num_classes])) + self.bias([self.num_classes])

        prediction = tf.nn.softmax(logits)
        pred = tf.nn.dropout(prediction, keep_prob=1.0)
        # 定义损失函数
        cross_entropy = -tf.reduce_sum(y * tf.log(pred))
        # 使用梯度下降进行优化
        tf.add_to_collection('loss', cross_entropy)
        loss_op = tf.add_n(tf.get_collection('loss'))

        train_op = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss_op)

        # 结果存放在一个布尔型列表中
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # 求准确率
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        cul = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for k in range(self.epochs):
                for item in range(9):
                    batch_x = X_train[batch_size * (item): batch_size * (item + 1)]
                    batch_y = y_train[batch_size * (item): batch_size * (item + 1)]
                    batch_x = batch_x.reshape((-1, self.timesteps, self.num_input))
                    sess.run(self.accuracy, feed_dict={x: batch_x, y: batch_y, learning_rate: self.lr})
                    sess.run(train_op, feed_dict={x: batch_x, y: batch_y, learning_rate: self.lr})

                    train_acc = self.accuracy.eval(session=sess, feed_dict={x: batch_x,y: batch_y,learning_rate: self.lr})

                batch_y = y_train
                #batch_x = batch_x.reshape((-1, self.timesteps, self.num_input))
                batch_x = X_train.reshape((-1, self.timesteps, self.num_input))
                l = sess.run(loss_op, feed_dict={x: batch_x,
                                                  y: batch_y,
                                                  learning_rate: self.lr})

                #acc = sess.run(accuracy, feed_dict={x: batch_x, y: y_test, learning_rate: self.lr})

                batch_x = X_train.reshape((-1, self.timesteps, self.num_input))
                batch_y = y_train
                l0 = sess.run(loss_op, feed_dict={x: batch_x,y: batch_y,learning_rate: self.lr})

                acc0 = sess.run(self.accuracy, feed_dict={x: batch_x, y: batch_y})

                cul.append(acc0)

            X_test = X_test.reshape(-1, self.timesteps, self.num_input)
            X_train = X_train.reshape(-1, self.timesteps, self.num_input)
            #X_test1 = X_test1.reshape(-1, self.timesteps, self.num_input)#######

            acc = sess.run(self.accuracy, feed_dict={x: X_test, y: y_test})
            print("acc:"+str(acc))

            self.score = acc
            self.predict_proba = sess.run(pred, feed_dict={x: X_test, y: y_test})
            self.cul = cul
            print("cul:"+str(cul))
            f = open("./out/036BiLstm.txt","a+")
            f.write("\n####################################\n")
            f.write("acc:"+str(acc)+"\ncul:\n"+str(cul))
            f.close()
        sess.close()


    def score_(self, X_test, y_test):
        return 0

    def prediction(self):
        predict = []
        for i in self.predict_proba:
            if i[0] > i[1]:
                predict.append(1)
            else:
                predict.append(0)
        return predict

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
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    #model = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)  ######################################################

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

X, y = load_file()

cv = StratifiedKFold(n_splits=10)
acc = []
for i, (train, test) in enumerate(cv.split(X, y)):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train])
    X_test = scaler.transform(X[test])
    y_train = trans(y[train])
    y_test = trans(y[test])
    x_shape= X.shape[0]
    x_data_shape= X.shape[1]

    lstm = BiLSTM(lr=0.001, timesteps=96, num_input=200, batch_size=50, num_hidden=128, epochs=100, lamda=0.01, i=0)
    lstm.clf(X_train,y_train,X_test,y_test)