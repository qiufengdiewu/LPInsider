# -*- coding: utf-8 -*-
import pandas as pd
import nltk
data=pd.read_csv('./017interaction.txt',sep='\t',header=None)
interactions=[]
for i in range(len(data)):
    interactions.append(data[0][i])
print(interactions)
data=pd.read_csv('./017neg.txt',sep='\t',header=None)
neg=[]
for i in range(len(data)):
    neg.append(data[0][i])
print(neg)
data = pd.read_csv('./out/0161Abstract_lemmatize_sentence.txt', sep='\t', header=None)
f=open("./out/017Abstract_classification_exist_interaction_neg.txt","w")
f1=open("./out/017Abstract_classification_without_interaction_and_neg.txt","w")
for i in range(len(data)):
    str=data[0][i]
    tokens = nltk.word_tokenize(str)
    #print tokens
    exist_interaction_str=""
    exist_interaction=False
    for j in tokens:
        if j in interactions:
            exist_interaction_str=j
            exist_interaction=True
            break

    exist_neg=False
    exist_neg_str=""
    for j in tokens:
        if j in neg:
            exist_neg = True
            exist_neg_str=j
            break

    if exist_neg==True and exist_interaction==True :
        f.write(exist_interaction_str+"\t"+exist_neg_str+'\t'+str+"\n")
        #print ("exist")
    else:
        f1.write(str+"\n")
        #print ("not")

f.close()
f1.close()