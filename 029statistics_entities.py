# coding=utf-8
import pandas as pd
if __name__ == '__main__':
    lncRNA_set = set()
    lncRNA_HGNC = open("./out/Collection of LncRNA and protein library/lncRNAs_HGNC.txt", mode="r")
    lncRNAs = lncRNA_HGNC.readlines()
    for lncRNA in lncRNAs:
        lncRNA = lncRNA[0:len(lncRNA)-1]
        lncRNA_set.add(lncRNA)
    lncRNA_LncRInter = open("./out/Collection of LncRNA and protein library/lncRNAs_LncRInter.txt", mode="r")
    lncRNAs = lncRNA_LncRInter.readlines()
    for lncRNA in lncRNAs:
        lncRNA = lncRNA[0:len(lncRNA) - 1]
        lncRNA_set.add(lncRNA)

    lncRNA_RAID = open("./out/Collection of LncRNA and protein library/lncRNAs_RAID.txt",mode="r")
    lncRNAs = lncRNA_RAID.readlines()
    for lncRNA in lncRNAs:
        lncRNA = lncRNA[0:len(lncRNA) - 1]
        lncRNA_set.add(lncRNA)
    print("length of lncRNA:"+str(len(lncRNA_set)))

    protein_set = set()
    protein_LncRInter = open("./out/Collection of LncRNA and protein library/proteins_LncRInter.txt", mode="r")
    proteins = protein_LncRInter.readlines()
    print(len(proteins))
    for protein in proteins:
        protein_set.add(protein[0:len(protein)-1])

    proteins_RAID = open("./out/Collection of LncRNA and protein library/proteins_RAID.txt",mode="r")
    proteins = proteins_RAID.readlines()
    print(len(proteins))
    for protein in proteins:
        protein_set.add(protein[0:len(protein)-1])

    proteins_uniprot = open("./out/Collection of LncRNA and protein library/proteins_uniprot_set.txt", mode="r")
    proteins = proteins_uniprot.readlines()
    print(len(proteins))
    for protein in proteins:
        protein_set.add(protein[0:len(protein) - 1])

    print("length of protein:" + str(len(protein_set)))