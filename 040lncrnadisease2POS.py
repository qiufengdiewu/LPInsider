# coding=utf-8
import stanfordcorenlp
import pandas as pd
def process(sentence):
    sentence = str(sentence).replace(",","").replace(".","").replace("(","").replace(")","").replace("'","")
    sentence = sentence.replace("'","").replace("-","").replace(":","")
    return sentence
if __name__ == '__main__':
    lncrnadisease = pd.read_csv("./in/lncRNAdisease.txt", sep="\t", header=None)
    lncRNA_HGNC = pd.read_csv("./out/Collection of LncRNA and protein library/lncRNAs_HGNC.txt",sep="\t",header=None)
    lncRNA_LncRInter = pd.read_csv("./out/Collection of LncRNA and protein library/lncRNAs_LncRInter.txt",sep="\t",header=None)
    lncRNA_RAID = pd.read_csv("./out/Collection of LncRNA and protein library/lncRNAs_RAID.txt",sep="\t",header=None)
    lncrna_set = set()
    for i in range(len(lncRNA_HGNC)):
        lncrna_set.add(lncRNA_HGNC[0][i])
    for i in range(len(lncRNA_LncRInter)):
        lncrna_set.add(lncRNA_LncRInter[0][i])
    for i in range(len(lncRNA_RAID)):
        lncrna_set.add(lncRNA_RAID[0][i])
    sentence_set = set()
    for i in range(len(lncrnadisease)):
        relationship = str(lncrnadisease[3][i]).lower()
        if relationship == "RNA-Protein".lower():
            lncrna_set.add(lncrnadisease[1][i])
            sentence = lncrnadisease[5][i]
            sentence_set.add(sentence)

    path = 'I:/stanford_parser/stanford-corenlp-full-2018-10-05'
    nlp = stanfordcorenlp.StanfordCoreNLP(path)
    f =open("./out/040lncrnadisease2POS.txt","w")
    for sentence in sentence_set:
        sentence = process(sentence)
        words_pos = nlp.pos_tag(sentence)
        for i in range(len(words_pos)):
            word = words_pos[i][0]
            pos = words_pos[i][1]
            if word not in lncrna_set:
                f.write(word+"\t"+pos+"\n")
            else:
                f.write(word+"\tlncRNA\n")

    f.close()