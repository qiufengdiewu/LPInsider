# coding:utf-8
import pandas as pd
import numpy as np
import gensim
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
import joblib
from sklearn.linear_model import LogisticRegression
# 对输入的文本进行预处理，并分词。
def datapreprocess(sent):
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
    temp_words = []
    for word in words:
        if len(word) > 0:
            temp_words.append(word)
    words = temp_words
    return words

#获取句子的语义词向量
def get_Semantic_word_vector(size, npLength, words, model, model_train):
    vec = []
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

    while npLength > 0:
        vec = np.append(vec, np.zeros(size).reshape(1, size))
        npLength -= 1
    return vec


def get_entity_position_vector(words,entity,entity_location,model,model_train, npLength):
    entity_w2v = 0
    matrix = np.zeros((1, npLength))
    entity_matrix = matrix[0]
    try:
        entity_w2v = model[entity]
    except:
        try:
            entity_w2v = model_train[entity]
        except:
            pass
    count = 0
    for i in range(entity_location-1, -1, -1):
        try:
            word_w2v = model[words[i]]
            entity_matrix[entity_location-1 - count] = pairwise_distances([entity_w2v, word_w2v])[0][1]
            count += 1
            if count >= 3:
                break
        except:
            try:
                word_w2v = model_train[words[i]]
                entity_matrix[entity_location-1 - count] = pairwise_distances([entity_w2v, word_w2v])[0][1]
                count += 1
                if count >= 3:
                    break
            except:
                pass
    count = 0
    for i in range(entity_location + 1, len(words)):
        try:
            word_w2v = model[words[i]]
            entity_matrix[entity_location+ 1 + count] = pairwise_distances([entity_w2v, word_w2v])[0][1]
            count += 1
            if count >= 3:
                break
        except:
            try:
                word_w2v = model_train[words[i]]
                entity_matrix[entity_location+ 1 - count] = pairwise_distances([entity_w2v, word_w2v])[0][1]
                count += 1
                if count >= 3:
                    break
            except:
                pass
    return entity_matrix

# 计算位置特征;返回lncRNA_matrix和protein_matrix
def get_position_vector(words, lncRNA, protein, model,model_train, npLength):
    try:
        lncRNA_location = words.index(lncRNA)
    except:
        lncRNA_location = -1
    try:
        protein_location = words.index(protein)
    except:
        protein_location = -1
    lncRNA_matrix = get_entity_position_vector(words, lncRNA, lncRNA_location, model, model_train, npLength)
    protein_matrix = get_entity_position_vector(words, protein, protein_location, model, model_train,npLength)
    return lncRNA_matrix, protein_matrix

# 包括计算对应的位置特征
def get_sent_vec(size, npLength, sent, model, model_train, lncRNA,protein,length_POS,sent_POS,POS_classified0,POS_matrix,length_classified):
    #预处理
    words = datapreprocess(sent)
    #获取语义词向量
    vec = get_Semantic_word_vector(size,npLength,words,model,model_train)
    # 计算位置特征
    lncRNA_matrix, protein_matrix = get_position_vector(words, lncRNA, protein, model, model_train,npLength)

    ######计算词性特征
    vec_POS=[]
    words_POS=str(sent_POS).split(" ")
    temp_words_POS = []
    for word_POS in words_POS:
        if len(word_POS)>0:
            temp_words_POS.append(word_POS)
    words_POS = temp_words_POS

    for word_POS in words_POS:
        for i in range(length_classified):
            if str(word_POS)==str(POS_classified0[i]):
                vec_POS=np.append(vec_POS,POS_matrix[i])
                length_POS-=1
                break
    while length_POS > 0:
        vec_POS = np.append(vec_POS,np.zeros(length_classified).reshape(1,length_classified))
        length_POS -= 1
    #####################

    vec = nomalization(vec)
    vec = vec.reshape([int(len(vec)/size),size])
    lncRNA_matrix = nomalization(lncRNA_matrix)
    protein_matrix = nomalization(protein_matrix)
    lncRNA_matrix = lncRNA_matrix.reshape([len(lncRNA_matrix), 1])
    protein_matrix = protein_matrix.reshape([len(protein_matrix), 1])
    vec_POS = nomalization(vec_POS)
    vec_POS = vec_POS.reshape([int(len(vec_POS)/11),11])

    temp_vec=[]
    for i in range(len(vec)):
        temp = np.concatenate((vec[i],lncRNA_matrix[i],protein_matrix[i],vec_POS[i]), axis=0)
        temp_vec.append(temp)
    vec = np.concatenate(temp_vec, axis= 0)
    return vec


# 训练模型

X = pd.read_csv("./out/007X_with_entity_and_stanford_parser_preprocess.txt", sep='\t', header=None,
                encoding='ISO-8859-1')  #######
_025POS_transform_to_unite = pd.read_csv("./in/025POS_transform_to_unite_preprocess.txt",sep="\t",header=None,encoding="utf-8")
y = np.load('./out/007X_with_entity_and_stanford_parser_preprocess.npy')  ####


def train(X, y):
    # 导入模型
    word2vec_path = "E:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
    #word2vec_path_train = './out/023Word2vec_model_modified_by_wiki'
    word2vec_path_train = './out/023Word2vec_model_modified_by_wiki_can_update_of_during_subsequent_training'
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    model_train = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)
    #model = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)
    for c in range(1):###################
        sentX = []
        sentX_POS=[]
        length = 0
        length_POS=0
        for i in range(0, len(X), 1):
            sentX.append(X[2][i])
            for sent in sentX:
                words = datapreprocess(sent)
                if len(words) > length:
                    length = len(words)
        print("length"+str(length))

        for i in range(len(_025POS_transform_to_unite)):
            sentX_POS.append(_025POS_transform_to_unite[2][i])
            for sent in sentX_POS:
                words = datapreprocess(sent)
                if len(words) > length_POS:
                    length_POS = len(words)

        print("length_POS" + str(length_POS))
        lncRNAs = []
        proteins = []
        for i in range(len(X)):
            lncRNAs.append(X[0][i])
            proteins.append(X[1][i])
        XX = []
        '''
        leng1=len(sentX)
        leng2=len(sentX_POS)
        #此时leng1和leng2相等，只需要用一个就OK了
        '''
        ############计算词性矩阵，例如NN：[0,1,0,0,0,0,0,0,0,0,0]
        POS_classified = pd.read_csv("./in/POS_classified.txt", sep='\t', header=None)
        length_classified = len(POS_classified)
        POS_classified0 = POS_classified[0]
        POS_matrix = np.zeros((length_classified,length_classified))
        for i in range(length_classified):
            POS_matrix[i][i] = 1


        ###############
        for i in range(len(sentX)):
            sent = sentX[i]
            sent_POS = sentX_POS[i]
            XX.append([get_sent_vec(200, length, sent, model, model_train,X[0][i],X[1][i],length_POS,sent_POS,POS_classified0,POS_matrix,length_classified)])
            i += 1

        XX = np.concatenate(XX)
        LogR = LogisticRegression(tol=0.001, solver='lbfgs', n_jobs=-1)
        LogR.fit(XX, y)

        print(LogR.predict(XX))
        joblib.dump(LogR, './out/LRmodel.pkl')
        print("joblib.dump(reg, './out/LRmodel.pkl')")

def nomalization(X):
    return preprocessing.scale(X, axis=0)

def datapreprocess(sent):
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
    temp_words = []
    for word in words:
        if len(word) > 0:
            temp_words.append(word)
    words = temp_words
    return words


if __name__ == '__main__':
    train(X, y)


