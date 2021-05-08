# coding=utf-8
import pandas as pd

raid= pd.read_csv("I:/RAID_V2/raid_V2.txt",sep="\t",header=None,low_memory=False)
Interactors_lncRNA=set()
Interactors_protein=set()


for i in range(1,len(raid)):
    interactor=raid[1][i]
    category=raid[2][i]
    if (str(category).lower()=="lncRNA".lower()):
        Interactors_lncRNA.add(interactor)
    elif (str(category).lower()=="protein".lower()):
        Interactors_protein.add(interactor)

    interactor = raid[5][i]
    category = raid[6][i]
    if (str(category).lower() == "lncRNA".lower()):
        Interactors_lncRNA.add(interactor)
    elif (str(category).lower() == "protein".lower()):
        Interactors_protein.add(interactor)


f_lncRNA=open("./out/027raid2Interactor_lncRNA.txt","w")
f_protein=open("./out/027raid2Interactor_protein.txt","w")

for lncRNA in Interactors_lncRNA:
    f_lncRNA.write(str(lncRNA) + "\tlncRNA\n")

for protein in Interactors_protein:
    f_protein.write(str(protein)+"\tprotein\n")


f_lncRNA.close()
f_protein.close()
