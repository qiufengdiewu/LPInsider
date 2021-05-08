# -*- coding: utf-8 -*-
import stanfordcorenlp
import pandas as pd

path = 'I:/stanford_parser/stanford-corenlp-full-2018-10-05'
nlp = stanfordcorenlp.StanfordCoreNLP(path)

sample = pd.read_csv('./in/neg_sample.txt', sep='\t', header=None)

f = open("./in/023neg_sample.txt", "w", encoding='UTF-8')

for i in range(len(sample)):
    lncRNA = str(sample[0][i])
    protein = str(sample[1][i])
    description = str(sample[2][i])  ########################################################
    description = description.replace('(', ' ')
    description = description.replace(',', ' ')
    description = description.replace(')', ' ')
    description = description.replace('.', ' ')
    description = description.replace("'", ' ')
    description = description.replace(':', ' ')
    description = description.replace('[', ' ')
    description = description.replace(']', ' ')
    description = description.replace('/', ' ')
    sentence = description

    print(sentence)
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

    print(description)
    print()
    write_context = str(lncRNA) + '\t' + str(protein) + '\t' + str(description) + '\n'
    f.write(write_context)
f.close()
nlp.close()
