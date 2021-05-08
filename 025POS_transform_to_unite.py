import pandas as pd
POS_classified=pd.read_csv("./in/POS_classified.txt",sep='\t',header=None)
length_classified=len(POS_classified)
POS_classified0=POS_classified[0]
POS_classified1=POS_classified[1]
for i in range(length_classified-1):
    POS_classified1[i]=POS_classified1[i].split(" ")
sample_POS=pd.read_csv("./in/023sample_word_transform_to_POS.txt",sep='\t',header=None)
f=open("./in/025POS_transform_to_unite.txt","w",encoding="utf-8")
for i in range(len(sample_POS)):
    lncRNA = sample_POS[0][i]
    protein = sample_POS[1][i]
    POS = sample_POS[2][i]
    POSs=POS.split(" ")
    POS_unite=""
    for k in range(len(POSs)):
        word=POSs[k]
        temp_unite=""
        for j in range(length_classified-1):
            flag=0
            for m in range(len(POS_classified1[j])):
                if word == POS_classified1[j][m]:
                    temp_unite=POS_classified0[j]
                    flag=1
                    break
                else:
                    temp_unite=POS_classified0[length_classified-1]
            if flag==1:
                break
        POS_unite+=(temp_unite+" ")
    f.write(lncRNA+"\t"+protein+"\t"+POS_unite+"\n")
f.close()