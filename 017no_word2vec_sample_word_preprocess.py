#coding:utf-8
import gensim
import pandas as pd
#读入数据
pos = pd.read_csv('./in/023pos_sample_preprocess.txt',sep='\t',header=None)
neg = pd.read_csv('./in/023neg_sample_preprocess.txt',sep='\t',header=None)
#导入模型
word2vec_path="I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path,binary=True)

def judge(sample):
    words_not_in=set()
    for i in range(len(sample)):
        description = str(sample[2][i])
        description = description.replace('(', ' ')
        description = description.replace(',', ' ')
        description = description.replace(')', ' ')
        description = description.replace('.', ' ')
        description = description.replace("'", ' ')
        description = description.replace(':', ' ')
        description = description.replace('[', ' ')
        description = description.replace(']', ' ')
        description = description.replace('/', ' ')
        descriptionD = description.split(" ")
        f_in = open("./out/017in_w2v_preprocess.txt", "a+")
        for word in descriptionD:
            try:
                model[str(word)]
                f_in.write(str(word)+" ")
            except:
                if len(str(word)) > 0:
                    words_not_in.add(str(word))
        f_in.write("\n")
        f_in.close()
    f = open("./out/017not_in_w2v_preprocess.txt", "a+")
    for word in words_not_in:
        f.write(str(word) + '\n')
    f.close()

judge(pos)
judge(neg)