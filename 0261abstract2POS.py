# coding=utf-8
import stanfordcorenlp
import pandas as pd


def process(sentence):
    try:
        if len(sentence) > 1:
            sentence = sentence.replace('(', '')
            sentence = sentence.replace(',', ' ')
            sentence = sentence.replace(')', '')
            sentence = sentence.replace('.', ' ')
            sentence = sentence.replace("'", ' ')
            sentence = sentence.replace(':', ' ')
            sentence = sentence.replace('[', '')
            sentence = sentence.replace(']', '')
            sentence = sentence.replace('/', ' ')
    except:
        print(sentence)
    return sentence


path = 'I:/stanford_parser/stanford-corenlp-full-2018-10-05'
nlp = stanfordcorenlp.StanfordCoreNLP(path)
lncRNA_abstract = pd.read_csv("./out/lncRNA_abstract.txt", sep="\t", header=None)
f = open("./out/0261abstract2POS.txt", "w", encoding='utf-8')

for i in range(len(lncRNA_abstract)):
    title = lncRNA_abstract[2][i]
    title = process(title)
    abstract = lncRNA_abstract[3][i]
    abstract = process(abstract)

    title = title.split(" ")
    if type(abstract) != type(0.1):
        abstract = abstract.split(" ")
    title_POS = []
    abstract_POS = []
    for j in range(len(title)):
        pos = nlp.pos_tag(title[j])
        title_POS.append(pos)
    if type(abstract) != type(0.1):
        for j in range(len(abstract)):
            pos = nlp.pos_tag(abstract[j])
            abstract_POS.append(pos)

    title_POSs = ""
    abstract_POSs = ""
    for j in range(len(title_POS)):
        try:
            if title_POS[j] != []:
                title_POSs += str(title_POS[j][0][1]) + " "
        except:
            print(title_POS[j])
            print(j)
    if len(abstract_POS) > 1:
        for j in range(len(abstract_POS)):
            try:
                if abstract_POS[j] != []:
                    abstract_POSs += str(abstract_POS[j][0][1]) + " "
            except:
                print(abstract_POS[j])
                print(j)

    string = title_POSs + "\t" + abstract_POSs + "\n"
    f.write(string)

f.close()
