#coding:utf-8
import gensim
import pandas as pd

#读入数据
#pos=pd.read_csv('./in/neg_sample.txt',sep='\t',header=None)
pos=pd.read_csv('./out/LncRInter_TT.txt',sep='\t',header=None)


f=open("017not_in_w2v_pos.txt","w")

#导入模型
word2vec_path="I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
model= gensim.models.KeyedVectors.load_word2vec_format(word2vec_path,binary=True)


count_LncRNA=0
count_protein=0
for i in range(len(pos)):
    f.write(str(i+1)+'\t')
    LncRNA = str(pos[0][i])
    protein = str(pos[1][i])

    try:
        model[str(LncRNA)]
    except:
        f.write(str(LncRNA))
        count_LncRNA+=1

    try:
        model[str(protein)]
    except:
        f.write('\t'+str(protein))
        count_protein+=0

    f.write('\n')
    '''
    description = str(pos[3][i])
    description = description.replace(",", " ")
    description = description.replace(".", " ")
    description = description.replace(";", " ")
    description = description.replace("(", " ")
    description = description.replace(")", " ")
    descriptionD = description.split(" ")
    print descriptionD

    flag = 0
    for word in descriptionD:
        count+=1
        try:
            print "word:" + str(word)
            model[str(word)]
        except:
            count_not_in_context+=1
            f.write(word + " ")
            flag = 1
    if flag == 1:
        f.write("\n")
    '''
    """
    LncRNA=str(pos[0][i])
    protein=str(pos[1][i])
    try:
        print "LncRNA"+str(LncRNA)
        model[str(LncRNA)]
    except:
        print("not in model")
        LncRNA=str(i+1)+" "+str(LncRNA)+"****"
        f.write(LncRNA)

    try:
        print "protein" + str(protein)
        model[str(protein)]
    except:
        print("not in model")
        protein = str(i + 1) + " " + str(protein) + "!!!!!"
        f.write(protein)
    f.write("\n")
    """
f.write(str(count_LncRNA)+'\n')
f.write(str(count_protein)+'\n')
#f.write(str(count)+'\n\n')
#f.write(str(count_not_in_context)+'\n')
f.close()