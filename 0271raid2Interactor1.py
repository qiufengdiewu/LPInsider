# coding=utf-8
import pandas as pd
import stanfordcorenlp
path = 'I:/stanford_parser/stanford-corenlp-full-2018-10-05'
nlp = stanfordcorenlp.StanfordCoreNLP(path)

abstract = pd.read_csv("./in/pos_neg_sample.txt",sep="\t",header=None,low_memory=False)
f=open("./out/027train_NER_model_words.txt","w")

entity = pd.read_csv("./out/027raid2Interactor1.txt",sep="\t",header=None,low_memory=False)
entity_lncRNA=set()
entity_protein=set()
for i in range(len(entity)):
    if str(entity[1][i])=="lncRNA":
        entity_lncRNA.add(str(entity[0][i]))
    else:
        entity_protein.add(str(entity[0][i]))
print(entity_lncRNA)
print(entity_protein)
print("h19" in entity_lncRNA)
print("H19" in entity_lncRNA)
entity_lncRNA
Interactors=[]
Categorys=[]
for i in range(len(abstract)):
    lncRNA=str(abstract[0][i])
    protein=str(abstract[1][i])
    f.write(str(lncRNA)+"\tlncRNA\n")
    f.write(str(protein)+"\tprotein\n")
    sentence=str(abstract[2][i])
    pos__tag = nlp.pos_tag(sentence)

    for j in range(len(pos__tag)):
        word=pos__tag[j][0]
        pos_=pos__tag[j][1]
        if word.lower() == lncRNA.lower():
            f.write(str(word) + "\tlncRNA\n")
        elif word.lower() == protein.lower():
            f.write(str(word) + "\tprotein\n")
        elif word in entity_lncRNA:
            f.write(str(word) + "\tlncRNA\n")
        elif word in entity_protein:
            f.write(str(word) + "\tprotein\n")
        else:
            f.write(str(word) + "\t"+str(pos_)+"\n")

f.close()

'''
下面是使用raid的内容提取出lncRNA和protein，但是在Stanford NER中的效果不好，决定先暂时放弃使用这种方法。
'''
'''
raid= pd.read_csv("I:/RAID_V2/raid_V2.txt",sep="\t",header=None,low_memory=False)
Interactors=[]
Categorys=[]
for i in range(1,len(raid)):
    interactor=raid[1][i]
    category=raid[2][i]
    if (str(category).lower()=="lncRNA".lower() or str(category).lower()=="protein".lower() and (interactor not in Interactors)):

        Interactors.append(interactor)
        Categorys.append(category)

    interactor = raid[5][i]
    category = raid[6][i]
    if (str(category).lower()=="lncRNA".lower() or str(category).lower()=="protein".lower() and (interactor not in Interactors)):
        Interactors.append(interactor)
        Categorys.append(category)


f=open("./out/027raid2Interactor1.txt","w")
for i in range(len(Interactors)):
    print(str(Interactors[i])+"\t"+str(Categorys[i]))
    f.write(str(Interactors[i])+"\t"+str(Categorys[i])+"\n")
f.close()

'''