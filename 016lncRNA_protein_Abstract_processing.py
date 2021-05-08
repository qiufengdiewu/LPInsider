# -*- coding: utf-8 -*-
import pandas as pd
import string
import re
import numpy as np

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer


def get_stopwords():
    stopwords = []
    with open('./in/stopwords.txt', 'r') as f:
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


# if __name__ == "__main__":
def main():
    data = pd.read_csv('./out/015lncRNA_protein_abstract_together.txt', sep='\t', header=None)

    stopwords = get_stopwords()
    num = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')
    del_ch = string.punctuation
    del_ch = del_ch.replace('-', '')
    del_ch = del_ch.replace('/', '')
    f = open('./out/016datap.txt', 'w')
    for i in range(0, len(data)):
        pmid=data[0][i]
        year=data[1][i]
        title=data[3][i]
        s = data[5][i]
        if (i % 100 == 0):
            print("data processing,----" + str(i) + '/' + str(len(data)) + "----")

        if (type(title)==float):
            if np.isnan(title):
                continue
        if (type(s) == float):
            if np.isnan(s):
                continue
        title=title.lower()
        title=re.sub(r'[^\x00-\x7f]', ' ', s)  # 去除非ASCII
        title=title.replace('/', ' ')
        title=lemmatize_sentence(title)
        '''for j in stopwords:  # 去除常用词
            while j in title:
                title.remove(j)
        '''


        s = s.lower()  # 全部转成小写字母
        s = re.sub(r'[^\x00-\x7f]', ' ', s)  # 去除非ASCII
        s = s.replace('/', ' ')
        s = lemmatize_sentence(s)
        '''
        for j in del_ch:  # 去除标点符号
            while j in s:
                s.remove(j)
        for j in stopwords:  # 去除常用词
            while j in s:
                s.remove(j)
        
        for j in num:
            while j in s:
                s.remove(j)'''
        f.write(str(pmid)+'\t'+str(year)+'\t')
        for j in title:
            f.write(str(j)+' ')
        f.write('\t')
        for j in s:
            f.write(str(j) + ' ')
        f.write('\t\n')

    f.close()


if __name__ == "__main__":
    main()