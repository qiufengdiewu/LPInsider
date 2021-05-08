# coding=utf-8
import pandas as pd
RAID=pd.read_csv("I:/RAID_V2/raid.v2_all_data.txt",sep="\t",header=None,low_memory=False)
entities_032=pd.read_csv("./out/032crawl_entities.txt",sep="\t",header=None)
lncRNA_032=[]
protein_032=[]
for i in range(len(entities_032)):
    rna=str(entities_032[1][i])
    if rna not in lncRNA_032:
        lncRNA_032.append(rna)
    prot=str(entities_032[3][i])
    if prot not in protein_032:
        protein_032.append(prot)

RAID_length=len(RAID)
f_lncRNA=open("./out/032raid_lncRNAs.txt","w")
f_protein=open("./out/032raid_proteins.txt","w")

lncRNA_unique=[]
lncRNA_RAID_unique=[]
protein_unique=[]
protein_RAID_unique=[]
for i in range(RAID_length):
    RAID_id=str(RAID[0][i])
    Interactor1 = str(RAID[1][i])
    Category1=str(RAID[2][i]).lower()
    Interactor2 = str(RAID[5][i])
    Category2=str(RAID[6][i]).lower()

    if not((Category1 == 'lncrna'and Category2=='protein')or(Category2=='lncrna'and Category1=='protein')):
        if Category1 == 'lncrna':
            f_lncRNA.write(RAID_id+"\t"+Interactor1+"\n")
            if Interactor1 not in lncRNA_unique and Interactor1 not in lncRNA_032:
                lncRNA_unique.append(Interactor1)
                lncRNA_RAID_unique.append(RAID_id)
        elif Category1=='protein':
            f_protein.write(RAID_id+"\t"+Interactor1+"\n")
            if Interactor1 not in protein_unique and Interactor1 not in protein_032:
                protein_unique.append(Interactor1)
                protein_RAID_unique.append(RAID_id)

        if Category2 == 'lncrna':
            f_lncRNA.write(RAID_id+"\t"+Interactor2+"\n")
            if Interactor2 not in lncRNA_unique and Interactor2 not in lncRNA_032:
                lncRNA_unique.append(Interactor2)
                lncRNA_RAID_unique.append(RAID_id)
        elif Category2=='protein':
            f_protein.write(RAID_id+"\t"+Interactor2+"\n")
            if Interactor2 not in protein_unique and Interactor2 not in protein_032:
                protein_unique.append(Interactor2)
                protein_RAID_unique.append(RAID_id)

    if i%10000==0:
        print(i)

f_lncRNA.close()
f_protein.close()


f_lncRNA_unique=open("./out/032raid_lncRNAs_unique.txt","w")
f_protein_unique=open("./out/032raid_proteins_unique.txt","w")
for i in range(len(lncRNA_unique)):
    f_lncRNA_unique.write(str(lncRNA_RAID_unique[i])+"\t"+str(lncRNA_unique[i])+"\n")
for i in range(len(protein_unique)):
    f_protein_unique.write(str(protein_RAID_unique[i])+"\t"+str(protein_unique[i])+"\n")
f_lncRNA_unique.close()
f_protein_unique.close()