# -*- coding: UTF-8 -*-
txt=open('./out/miRNAAbstract.txt','r').read()
txt=str(txt)
words=txt.split()
count={}
for word in words:
    count[word]=count.get(word,0)+1
item=list(count.items())

item.sort(key=lambda  x:x[1],reverse=True)
for i in range(20):
    word,count=item[i]
    print("{0:<10}{1:>5}".format(word, count))