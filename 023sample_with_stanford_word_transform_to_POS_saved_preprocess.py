# -*- coding: utf-8 -*-
import stanfordcorenlp
import pandas as pd
path='I:/stanford_parser/stanford-corenlp-full-2018-10-05'
nlp=stanfordcorenlp.StanfordCoreNLP(path)

sample=pd.read_csv('./in/023pos_sample_stanford_parser_preprocess.txt',sep='\t',header=None)   #######################

f=open("./in/023pos_sample_word_transform_to_POS_preprocess.txt","w",encoding='UTF-8')########################

for i in range(len(sample)):
    lncRNA=str(sample[0][i])
    protein=str(sample[1][i])
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
    sentence=description

    sentence_pos= nlp.pos_tag(sentence)
    POSs=""
    for i in range(len(sentence_pos)):
        POSs +=str(sentence_pos[i][1])+" "
    print(sentence_pos)
    print(POSs)
    f.write(lncRNA+"\t"+protein+"\t")
    f.write(POSs+"\n")

f.close()
nlp.close()