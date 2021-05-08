# coding=utf-8
import pandas as pd
import stanfordcorenlp

def process(sentence):
    sentence = sentence.replace('(', ' ').replace(',', ' ').replace(')', ' ').replace('.', ' ').replace("'", ' ').replace(':', ' ').replace('[', ' ').replace(']', ' ').replace("\n","")
    sentence = str(sentence).replace("<InteractingKeyWord>","")
    sentence = sentence.replace("</InteractingKeyWord>","")
    sentence = sentence.replace(",","").replace('"',"")
    sentence = sentence.replace("</OtherProtein>","").replace("</HumanProtein>","")
    words = sentence.split(" ")
    protein_set = set()
    for i in range(len(words)):
        if words[i] == "<HumanProtein>" or words[i] == "<OtherProtein>":
            i = i+1
            protein_set.add(words[i])
    sentence = sentence.replace("<HumanProtein>", "").replace("<OtherProtein>", "").replace("/", "")
    return sentence, protein_set
def process1(sentence):
    sentence =  str(sentence).replace("?","").replace("|","").replace("-","").replace("+","")
    return sentence

if __name__ == '__main__':
    path = 'I:/stanford_parser/stanford-corenlp-full-2018-10-05'
    nlp = stanfordcorenlp.StanfordCoreNLP(path)

    f = open("./out/040IntAct2POS.txt","w")
    protein_LncRInter = pd.read_csv("./out/Collection of LncRNA and protein library/proteins_LncRInter.txt",sep="\t",header=None)
    protein_RAID = pd.read_csv("./out/Collection of LncRNA and protein library/proteins_RAID.txt", sep="\t", header=None)
    protein_uniprot = pd.read_csv("./out/Collection of LncRNA and protein library/proteins_uniprot_set.txt", sep="\t",header=None)
    proteins_set = set()
    for i in range(len(protein_LncRInter)):
        proteins_set.add(protein_LncRInter[0][i])
    for i in range(len(protein_RAID)):
        proteins_set.add(protein_RAID[0][i])
    for i in range(len(protein_uniprot)):
        proteins_set.add(protein_uniprot[0][i])

    IntAct = open("./in/IntAct.txt", "r")
    line = IntAct.readline()
    sentences_set = set()
    while line:
        line = IntAct.readline()
        sentence, proteins = process(line)
        sentences_set.add(sentence)
        for protien in proteins:
            proteins_set.add(protien)
    for sentence in sentences_set:
        words_pos = nlp.pos_tag(sentence)
        for i in range(len(words_pos)):
            word = words_pos[i][0]
            pos = words_pos[i][1]
            if word not in proteins_set:
                f.write(word+"\t"+pos+"\n")
            else:
                f.write(word+"\tprotein\n")

    f.close()
    IntAct.close()