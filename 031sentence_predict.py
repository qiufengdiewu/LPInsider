# coding=utf-8
import stanfordcorenlp
import pandas as pd
import joblib
import numpy as np
import gensim
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
import re
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import string
import nltk



def preprocess(sentence):
    sentence=str(sentence)
    sentence=sentence.lower()
    sentence = re.sub(r'[^\x00-\x7f]', ' ', sentence)  # 去除非ASCII
    sentence.replace(".", " ")
    sentence.replace(",", " ")
    sentence.replace("/", " ")
    sentence.replace("'", "")
    sentence_process = lemmatize_sentence(sentence)

    stopwords = get_stopwords()
    for j in stopwords:
        while j in sentence_process:
            sentence_process.remove(j)

    del_ch = string.punctuation
    del_ch = del_ch.replace('-', '-')
    del_ch = del_ch.replace('/', '')
    for j in del_ch:
        while j in sentence_process:
            sentence_process.remove(j)

    sentence=""
    for word in sentence_process:
        sentence+=(word+" ")
    return sentence

#将原来的句子转换成Stanford树
def stanfor_parser(sentence):
    path = 'I:/stanford_parser/stanford-corenlp-full-2018-10-05'
    nlp = stanfordcorenlp.StanfordCoreNLP(path)
    word_tokenize = nlp.word_tokenize(sentence)
    dependency_parse = nlp.dependency_parse(sentence)
    root_loc = 0
    for i in range(len(dependency_parse)):
        if dependency_parse[i][0] == str("ROOT").upper():
            root_loc = i
            break
    dependency_parse_dict = {}
    dependency_parse_dict[str(dependency_parse[root_loc][1])] = [str(dependency_parse[root_loc][2])]

    for i in range(len(dependency_parse)):
        if str(dependency_parse[i][1]) not in dependency_parse_dict:
            dependency_parse_dict[str(dependency_parse[i][1])] = [str(dependency_parse[i][2])]
        else:
            seq = dependency_parse_dict[str(dependency_parse[i][1])]
            if str(dependency_parse[i][2]) not in seq:
                seq.append(str(dependency_parse[i][2]))
            dependency_parse_dict[str(dependency_parse[i][1])] = seq

    # print(dependency_parse_dict)
    # 树的DFS遍历

    def DFS(graph, s, queue=[]):
        queue.append(s)
        try:
            for i in graph[s]:
                if i not in queue:
                    DFS(graph, i, queue)
        except:
            pass

        return queue

    dependency_parse_sorted = DFS(dependency_parse_dict, '0')
    description = ''
    for i in range(1, len(dependency_parse_sorted)):
        description += (word_tokenize[int(dependency_parse_sorted[i]) - 1] + ' ')
    nlp.close()
    return description


def sentence_predict(sentence,lncRNA,protein ,model,model_train):
    sentence=preprocess(sentence)
    sentence=stanfor_parser(sentence)
    # 导入模型
    word2vec_path = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
    # word2vec_path_train = './out/023Word2vec_model_modified_by_wiki'
    word2vec_path_train =  './out/023Word2vec_model_modified_by_wiki_can_update_of_during_subsequent_training'
    #model = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)  ######################################################
    model_POS = gensim.models.word2vec.Word2Vec.load("./out/024POS_of_words.model")
    length = 97
    length_POS = 97
    lncRNAs=[]
    proteins=[]

    ############计算词性矩阵，例如NN：[0,1,0,0,0,0,0,0,0,0,0]
    POS_classified = pd.read_csv("./in/POS_classified.txt", sep='\t', header=None)
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

    LGBM_model = joblib.load("./out/011xgboost_model.pkl")
    pred = LGBM_model.predict(XX)
    """
    if int(pred[0])==1:
        pred="正样本"
    else:
        pred="负样本"
    """
    return pred



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

    path_nlp = 'I:/stanford_parser/stanford-corenlp-full-2018-10-05'
    nlp = stanfordcorenlp.StanfordCoreNLP(path_nlp)
    sentence_pos=nlp.pos_tag(sentence)

    POSs=""
    for i in range(len(sentence_pos)):
        POSs+=str(sentence_pos[i][1])+" "

    sentence_pos=POSs.split(" ")

    POS_classified = pd.read_csv("./in/POS_classified.txt", sep='\t', header=None)
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
    nlp.close()
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
    vec=vec[0]
    vec=nomalization(vec)
    lncRNA_matrix=nomalization(lncRNA_matrix)
    protein_matrix=nomalization(protein_matrix)
    vec_POS=nomalization(vec_POS)
    vec=np.concatenate((vec,lncRNA_matrix,protein_matrix,vec_POS),axis=0)
    return nomalization(vec)




def nomalization(X):
    return preprocessing.scale(X, axis=0)


def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return res

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_stopwords():
    stopwords = []
    with open("./in/stopwords.txt","r") as f:
        for i in f.readlines():
            stopwords.append(i.strip())

    return stopwords

def splitSentence(paragraph):
   tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
   sentences = tokenizer.tokenize(paragraph)
   return sentences


if __name__ == "__main__":
    raid2entities_with_reference=pd.read_csv("./out/029raid2entities_with_reference.txt",sep="\t",header=None)
    raid_reference_abstract=pd.read_csv("./out/030raid_reference_abstract.txt",sep="\t",header=None)
    PMIDs=raid_reference_abstract[0]

    f=open("./out/031predict_result.txt","w")

    word2vec_path = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
    word2vec_path_train = './out/023Word2vec_model_modified_by_wiki_can_update_of_during_subsequent_training'
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    model_train = gensim.models.word2vec.Word2Vec.load(word2vec_path_train)

    raid2entities_length=len(raid2entities_with_reference)
    raid_reference_length=len(raid_reference_abstract)
    for i in range(raid2entities_length):
        lncRNA = str(raid2entities_with_reference[1][i]).lower()
        protein = str(raid2entities_with_reference[5][i]).lower()
        references = str(raid2entities_with_reference[11][i])
        references=references.split("//")
        for reference in references:
            for j in range(len(PMIDs)):
                if int(reference)== int(PMIDs[j]):#总数很少，目前还不用考虑算法的时间复杂度

                    abstract=str(raid_reference_abstract[2][j])
                    sentences = splitSentence(abstract)
                    for sentence in sentences:
                        sentence=str(sentence).lower()

                        if sentence.find(str(lncRNA))>=0 and sentence.find(str(protein))>=0:

                            result=sentence_predict(sentence, lncRNA, protein,model,model_train )
                            print(str(reference)+"\t"+str(lncRNA)+"\t"+str(protein)+"\t"+str(result)+"\t"+str(sentence)+"\t\n")
                            f.write(str(reference)+"\t"+str(lncRNA)+"\t"+str(protein)+"\t"+str(result)+"\t"+str(sentence)+"\t\n")

                    break

    f.close()
"""

sentence="miRNAs have been shown to be essential for normal cartilage development in the mouse. However, the role of specific miRNAs in cartilage function is unknown. Using rarely available healthy human chondrocytes (obtained from 8 to 50 year old patients), we detected a most highly abundant primary miRNA H19, whose expression was heavily dependent on cartilage master regulator SOX9. Across a range of murine tissues, expression of both H19- and H19-derived miR-675 mirrored that of cartilage-specific SOX9. miR-675 was shown to up-regulate the essential cartilage matrix component COL2A1, and overexpression of miR-675 rescued COL2A1 levels in H19 - or SOX9-depleted cells. We thus provide evidence that SOX9 positively regulates COL2A1 in human articular chondrocytes via a previously unreported miR-675-dependent mechanism. This represents a novel pathway regulating cartilage matrix production and identifies miR-675 as a promising new target for cartilage repair."
lncRNA="H19"
protein="COL2A1"

sentences=splitSentence(sentence)

for sentence in sentences:
    if lncRNA in sentence and protein in sentence:
        print(sentence)
        print(sentence_predict(sentence,lncRNA,protein))

"""