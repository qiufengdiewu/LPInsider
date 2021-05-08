RAID=open("I:/RAID_V2/raid.v2_all_data.txt","r")
line=RAID.readline()
entity_set=set()
while line:
    words=line.split("\t")
    if words[2].lower() == "lncrna":
        entity_set.add(words[1])
    if words[6].lower() == "lncrna":
        entity_set.add(words[5])

    line=RAID.readline()

entity_set=sorted(entity_set)
f=open("./out/999lncRNA.txt","w")#在这里修改lncRNA或者protein####################################
for entity in entity_set:
    f.write(str(entity)+"\n")
f.close()

'''
import pandas as pd
RAID=pd.read_csv("I:/RAID_V2/raid_V2.txt",sep="\t",header=None,low_memory=False)
RAID_length=len(RAID)
lncRNA_set=set()
for i in range(RAID_length):
    Category1=str(RAID[2][i])
    Category2=str(RAID[6][i])
    if str(Category1).lower()=="lncrna".lower():
        lncRNA_set.add(RAID[1])
        RAID[1]
    if str(Category2).lower()=="lncrna".lower():
        print(RAID[5])
        lncRNA_set.add(RAID[5])
print("++++++++++++")
print(lncRNA_set)

f=open("lncRNA.txt","w")
f.write(lncRNA_set)
f.close()
'''