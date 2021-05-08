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
    f = open('./out/0162Abstract_lemmatize_sentence.txt', 'w')
    for i in range(0, len(data)):
        abstract=lemmatize_sentence(str(data[5][i]))
        for j in abstract:
            f.write(str(j) + ' ')
        f.write("\t\n")
        if (i % 100 == 0):
            print("data processing,----" + str(i) + '/' + str(len(data)) + "----")
        '''
        pmid = data[0][i]
        s = data[5][i]
        if (i % 100 == 0):
            print("data processing,----" + str(i) + '/' + str(len(data)) + "----")
        flag=[]
        s=str(s).strip()

        #print s
        for j in range(len(s)):
            if (s[j]=='.'and j==(len(s)-1)or(s[j]=="."and s[j+1]==" "and s[j+2]>='A'and s[j+2]<'Z')):
                flag.append(j)
        txt=str(pmid)+"\toriginal:\t"+str(s)+"\t\n"
        f.write(txt)

        #print flag
        pre=0
        common_txt="\t\t"
        for k in flag:
            abstract_subsection = ""
            if pre==0:
                for L in s[pre:k]:
                    if L!='.':
                        abstract_subsection+=str(L)
                    pre=k
            else:
                for L in s[pre+1:k]:
                    if L!='.':
                        abstract_subsection+=str(L)
                    pre=k
            f.write(common_txt)
            abstract_subsection=lemmatize_sentence(abstract_subsection)
            for j in abstract_subsection:
                f.write(str(j) + ' ')
            f.write("\t\n")'''
    f.close()

if __name__ == "__main__":
    main()