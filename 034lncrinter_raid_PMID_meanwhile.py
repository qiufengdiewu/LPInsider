# coding=utf-8

import pandas as pd
lncRInter=pd.read_csv("./in/LncRInter.txt",sep="\t",header=None,low_memory=True)
raid_reference=pd.read_csv("./out/029raid2entities_with_reference.txt",sep="\t",header=None,low_memory=True)
lncRInter_PMID=set()
raid_reference_PMID=set()
pmid_meanwhile=set()
print(lncRInter)
for i in range(1,len(lncRInter)):
    if str(lncRInter[2][i]).lower()=="RNA-Protein".lower():
        lncRInter_PMID.add(int(lncRInter[9][i]))

for i in range(len(raid_reference)):
    try:
        pmid =int(raid_reference[11][i])
        if pmid in lncRInter_PMID :
            pmid_meanwhile.add(pmid)
    except:
        pmids=str(raid_reference[11][i]).split("//")
        for pmid in pmids:
            if pmid in lncRInter_PMID:
                pmid_meanwhile.add(pmid)

f=open("./out/034PMID_meanwhile.txt","w")
for pmid in pmid_meanwhile:
    f.write(str(pmid)+"\n")

f.close()