# -*- coding: utf-8 -*-
import stanfordcorenlp
import pandas as pd
path='I:/stanford_parser/stanford-corenlp-full-2018-10-05'
nlp=stanfordcorenlp.StanfordCoreNLP(path)

neg_sample=pd.read_csv('./in/023neg_sample_preprocess.txt',sep='\t',header=None)
pos_sample=pd.read_csv('./in/023pos_sample_preprocess.txt',sep='\t',header=None)

f_neg=open("./in/023neg_sample_stanford_parser_preprocess.txt","w",encoding='UTF-8')
f_pos=open("./in/023pos_sample_stanford_parser_preprocess.txt","w",encoding='UTF-8')


def process(sample,flag):
    for i in range(len(sample)):
        lncRNA = str(sample[0][i])
        protein = str(sample[1][i])
        sentence = str(sample[2][i])

        word_tokenize = nlp.word_tokenize(sentence)
        dependency_parse = nlp.dependency_parse(sentence)

        root_loc = 0
        for i in range(len(dependency_parse)):
            if dependency_parse[i][0] == str("ROOT").upper():
                root_loc = i
                break
        dependency_parse_dict = {}
        dependency_parse_dict[str(dependency_parse[root_loc][1])] = [str(dependency_parse[root_loc][2])]

        for i in range(len(dependency_parse)):
            if str(dependency_parse[i][1]) not in dependency_parse_dict:
                dependency_parse_dict[str(dependency_parse[i][1])] = [str(dependency_parse[i][2])]
            else:
                seq = dependency_parse_dict[str(dependency_parse[i][1])]
                if str(dependency_parse[i][2]) not in seq:
                    seq.append(str(dependency_parse[i][2]))
                dependency_parse_dict[str(dependency_parse[i][1])] = seq

        # print(dependency_parse_dict)
        # 树的DFS遍历

        def DFS(graph, s, queue=[]):
            queue.append(s)
            try:
                for i in graph[s]:
                    if i not in queue:
                        DFS(graph, i, queue)
            except:
                pass

            return queue

        dependency_parse_sorted = DFS(dependency_parse_dict, '0')
        description = ''
        for i in range(1, len(dependency_parse_sorted)):
            description += (word_tokenize[int(dependency_parse_sorted[i]) - 1] + ' ')

        write_context = str(lncRNA) + '\t' + str(protein) + '\t' + str(description) + '\t\n'
        if flag=="neg":
            f_neg.write(write_context)
        elif flag=="pos":
            f_pos.write(write_context)


process(neg_sample,"neg")
process(pos_sample,"pos")

f_neg.close()
f_pos.close()
nlp.close()


