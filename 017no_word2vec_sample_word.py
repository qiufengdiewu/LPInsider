#coding:utf-8
import gensim
import pandas as pd
import time
#读入数据
start = time.time()
pos=pd.read_csv('./in/neg_sample.txt',sep='\t',header=None)

f=open("017not_in_w2v_neg.txt","w")
f1=open("017in_w2v_neg.txt","w")
#导入模型
word2vec_path="I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
model= gensim.models.KeyedVectors.load_word2vec_format(word2vec_path,binary=True)


'''
应当去掉的符号：
(   ,  )    .   '   : [ ]
'''
count=0
count1=0
for i in range(len(pos)):
    description=str(pos[2][i])
    description = description.replace('(',' ')
    description = description.replace(',',' ')
    description = description.replace(')',' ')
    description = description.replace('.',' ')
    description = description.replace("'",' ')
    description = description.replace(':',' ')
    description = description.replace('[',' ')
    description = description.replace(']',' ')
    description = description.replace('/', ' ')
    descriptionD=description.split(" ")

    for word in descriptionD:
        try:
            model[str(word)]
            f1.write(str(word)+' ')
        except:
            if len(str(word))>0:
                f.write(str(word)+'\n')
                #print "write word:"+str(word)
                count1+=1
            print (str(len(str(word)))+':'+str(word))
            count+=1
    #f.write("\n")
    f1.write('\n')

f.close()
f1.close()
print (count)
print (count1)
end = time.time()
print (end - start, 's')