# -*- coding: utf-8 -*-
import pandas as pd
'''
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')'''
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


def main():
    data = pd.read_csv('./out/0161Abstract_subsection2.txt', sep='\t', header=None)

    f = open('./out/0161Abstract_lemmatize_sentence.txt', 'w')
    for i in range(len(data)):
        if (i % 100 == 0):
            print("data processing,----" + str(i) + '/' + str(len(data)) + "----")
        sentence=lemmatize_sentence(str(data[0][i]))
        for j in sentence:
            f.write(j+" ")
        f.write("\n")
    f.close()

if __name__ == "__main__":
    main()