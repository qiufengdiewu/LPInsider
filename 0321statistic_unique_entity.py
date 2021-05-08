# coding=utf-8
import pandas as pd

proteins = pd.read_csv("./out/0321crawl_protein.txt", sep="\t", header=None)
lncRNAs = pd.read_csv("./out/0321crawl_lncRNA.txt",sep="\t",header=None)
print(proteins)
print(lncRNAs)
lncRNA_number = 0
protein_number = 0
for i in range(len(lncRNAs)):
    lncRNA_number += 1
    alias = str(lncRNAs[2][i])
    alias = alias.split("|")
    for alia in alias:
        if len(alia) > 1:
            lncRNA_number += 1

for i in range(len(proteins)):
    protein_number += 1
    alias = str(proteins[2][i])
    alias = alias.split("|")
    for alia in alias:
        if len(alia)> 1 :
            protein_number += 1

f=open("./out/0321statistic_entities.txt","w")
f.write("lncRNA number:\t"+str(lncRNA_number)+"\nprotein number\t"+str(protein_number)+"\n")
f.close()