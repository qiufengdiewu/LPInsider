# coding:utf-8
import gensim
import pandas as pd

# 读入数据
pos = pd.read_csv('./in/neg_sample.txt', sep='\t', header=None)

f = open("./out/017not_in_w2v_entity_word.txt", "w")

# 导入模型
word2vec_path = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

for i in range(len(pos)):
    LncRNA = str(pos[0][i])
    protein = str(pos[1][i])
    flag = 0
    try:
        model[str(LncRNA)]
    except:
        flag=1
        print("no LncRNA w2v ")
        f.write(str(i+1))
        f.write(LncRNA)
    try:
        model[str(protein)]
    except:
        flag=1
        print ("no protein w2v")
        f.write(str(i+1))
        f.write(protein)
    if flag==1:
        f.write("\n")

f.close()