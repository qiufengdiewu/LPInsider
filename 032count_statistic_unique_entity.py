# coding=utf-8
import pandas as pd

entities = pd.read_csv("./out/032crawl_entities.txt", sep="\t", header=None)
entities_length = len(entities)
lncRNA = []
lncRNA_Alias = []
protein = []
protein_Alias = []
lncRNA_number = 0
protein_number = 0
for i in range(entities_length):
    if entities[1][i] not in lncRNA:
        lncRNA.append(entities[1][i])
        lncRNA_Alias.append(entities[2][i])
    if entities[3][i] not in protein:
        protein.append(entities[3][i])
        protein_Alias.append(entities[4][i])


f_lncRNA = open("./out/032crawl_lncRNA.txt","w")
if len(lncRNA)==len(lncRNA_Alias):
    for i in range(len(lncRNA)):
        lncRNA_number += 1
        alias=lncRNA_Alias[i]
        if alias!="-":
            alias=str(alias).replace("|","\t")
            alias+="\t"
        else:
            alias=""

        alias_split=alias.split("\t")

        for alia in alias_split:
            if len(alia)>1:
                lncRNA_number+=1
        f_lncRNA.write(str(lncRNA[i])+"\t"+alias+"\n")
f_lncRNA.close()

f_protein=open("./out/032crawl_protien.txt","w")
if len(protein_Alias) == len(protein):
    for i in range(len(protein)):
        protein_number += 1
        alias=str(protein_Alias[i])
        if alias!="-":
            alias=alias.replace("|","\t")
            alias+="\t"
        else:
            alias=""
        alias_split=alias.split("\t")

        for alia in alias_split:
            if len(alia)>1:
                protein_number+=1
        f_protein.write(str(protein[i])+"\t"+alias+"\n")
f_protein.close()

print(lncRNA_number)
print(protein_number)

f=open("./out/032statistic_entities.txt","w")
f.write("lncRNA number:\t"+str(lncRNA_number)+"\nprotein number:\t"+str(protein_number)+"\n")
f.close()
