# coding=utf-8

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#读入数据
temp_data = ['HSP90AB2P','AK294004','SNIP2']

#导入模型
word2vec_path = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v.bin"
word2vec_path_save = "I:/Word2vecModel/wikipedia-pubmed-and-PMC-w2v_save_word2vec_format.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

print(model)
model.save_word2vec_format(word2vec_path_save)
