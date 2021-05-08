# coding=utf-8
import pandas as pd
import re
import string
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

neg_sample=pd.read_csv('./in/neg_sample.txt',sep='\t',header=None)
pos_sample=pd.read_csv('./in/pos_sample.txt',sep='\t',header=None)

f_neg=open("./in/023neg_sample_preprocess.txt","w",encoding='UTF-8')
f_pos=open("./in/023pos_sample_preprocess.txt","w",encoding='UTF-8')


def preprocess(sample,loc,flag):
    for i in range(len(sample)):
        lncRNA = str(sample[0][i])
        protein = str(sample[1][i])
        description = str(sample[loc][i])
        description = description.replace('(', ' ')
        description = description.replace(',', ' ')
        description = description.replace(')', ' ')
        description = description.replace('.', ' ')
        description = description.replace("'", ' ')
        description = description.replace(':', ' ')
        description = description.replace('[', ' ')
        description = description.replace(']', ' ')
        description = description.replace('/', ' ')
        sentence_process = description

        sentence_process = re.sub(r'[^\x00-\x7f]', ' ', sentence_process)  # 去除非ASCII
        sentence_process.replace(".", " ")
        sentence_process.replace(",", " ")
        sentence_process.replace("/", " ")
        sentence_process.replace("'", "")
        sentence_process = lemmatize_sentence(sentence_process)
        '''
        stopwords = get_stopwords()
        for j in stopwords:
            if j in sentence_process:
                sentence_process.remove(j)
        '''
        del_ch = string.punctuation
        del_ch = del_ch.replace('-', '-')
        del_ch = del_ch.replace('/', '')
        for j in del_ch:
            if j in sentence_process:
                sentence_process.remove(j)
        sentence=""
        for i in range(len(sentence_process)):
            sentence+=(str(sentence_process[i])+" ")

        if flag=="neg":
            f_neg.write(lncRNA+"\t"+protein+"\t"+sentence+"\t\n")
        elif flag=="pos" :
            f_pos.write(lncRNA+"\t"+protein+"\t"+sentence+"\t\n")



def get_stopwords():
    stopwords = []
    with open("./in/stopwords.txt","r") as f:
        for i in f.readlines():
            stopwords.append(i.strip())

    return stopwords

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

def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return res


preprocess(neg_sample,int(2),"neg")
preprocess(pos_sample,int(3),"pos")
f_neg.close()
f_pos.close()