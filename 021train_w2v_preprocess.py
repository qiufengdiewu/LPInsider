# conding=utf-8
import csv

def remove_space(sentence):
    temp_sentence=""
    words=sentence.split(" ")
    for i in range(len(words)):
        if len(words[i])>0:
            temp_sentence+=words[i]
            if i<len(words)-1:
                temp_sentence+=" "

    return temp_sentence.strip()

def process(sentence):
    tags=[]
    for i in range(len(sentence)):
        tag = ""
        if sentence[i]=="<":
            locA=i
            for j in range(i,len(sentence)):
                if sentence[j]==">":
                    locB=j

                    for k in range(locA,locB+1):
                        tag+=sentence[k]
                    break
            tags.append(tag)
    for tag in tags:
        sentence=sentence.replace(tag,"")

    sentence = sentence.replace('(', '')
    sentence = sentence.replace(',', ' ')
    sentence = sentence.replace(')', '')
    sentence = sentence.replace('.', ' ')
    sentence = sentence.replace("'", ' ')
    sentence = sentence.replace(':', ' ')
    sentence = sentence.replace('[', '')
    sentence = sentence.replace(']', '')
    sentence = sentence.replace('/', ' ')

    sentence=remove_space(sentence)
    return sentence

f=open("./out/021train_w2v_preprocess.txt","w",encoding="utf-8")
with open("./in/AIMED_processed_all.csv","r",newline="",encoding="utf-8") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        sentence=line[3]
        sentence=process(sentence)
        f.write(str(sentence)+"\n")
f.close()
