# coding=utf-8
import stanfordcorenlp
import pandas as pd
"""
注释一是将lncrinter的正样本作为训练集

"""
def precess(description):
    description = description.replace('(', ' ')
    description = description.replace(',', ' ')
    description = description.replace(')', ' ')
    description = description.replace('.', ' ')
    description = description.replace("'", ' ')
    description = description.replace(':', ' ')
    description = description.replace('[', ' ')
    description = description.replace(']', ' ')
    description = description.replace('/', ' ')
    return description

#将正负样本转换成Stanford ner可以处理的格式
if __name__ == '__main__':
    path = 'I:/stanford_parser/stanford-corenlp-full-2018-10-05'
    nlp = stanfordcorenlp.StanfordCoreNLP(path)
    sample = pd.read_csv('./in/pos_neg_sample.txt', sep='\t', header=None)
    f = open("./out/039sample2wordPOS.txt","w")
    for i in range(len(sample)):
        lncRNA = str(sample[0][i])
        protein = str(sample[1][i])
        description = str(sample[2][i])
        sentence = precess(description)
        sentence_pos = nlp.pos_tag(sentence)
        for i in range(len(sentence_pos)):
            word = sentence_pos[i][0]
            if word == lncRNA:
                f.write(word +"\tlncRNA\n")
            elif word == protein:
                f.write(word+"\tprotein\n")
            else:
                pos = sentence_pos[i][1]
                f.write(word+"\t"+pos+"\n")
    nlp.close()



"""#注释一
if __name__ == '__main__':
    path = 'I:/stanford_parser/stanford-corenlp-full-2018-10-05'
    nlp = stanfordcorenlp.StanfordCoreNLP(path)
    sample = pd.read_csv('./in/pos_sample.txt', sep='\t', header=None)
    f = open("./out/039description2wordPOS.txt","w")
    for i in range(len(sample)):
        lncRNA = str(sample[0][i])
        protein = str(sample[1][i])
        description = str(sample[3][i])
        sentence = precess(description)
        sentence_pos = nlp.pos_tag(sentence)
        for i in range(len(sentence_pos)):
            word = sentence_pos[i][0]
            if word == lncRNA:
                f.write(word +"\tlncRNA\n")
            elif word == protein:
                f.write(word+"\tprotein\n")
            else:
                pos = sentence_pos[i][1]
                f.write(word+"\t"+pos+"\n")
    nlp.close()

"""
