# coding=utf-8
import pandas as pd
if __name__ == '__main__':
    RAID=pd.read_csv("I:/RAID_V2/raid.v2_all_data.txt",sep="\t",header=None,low_memory=False)
    RAID_length=len(RAID)
    f=open("I:/RAID_V2/raid2entities.txt","w")
    f1=open("./out/029raid2entities_with_reference.txt",'w')
    f_lncRNA = open("I:/RAID_V2/lncRNAs_RAID.txt","w")
    f_protein = open("I:/RAID_V2/proteins_RAID.txt","w")
    lncRNAs = set()
    proteins =set()
    for i in range(RAID_length):
        Category1 = str(RAID[2][i]).lower()
        Category2 = str(RAID[6][i]).lower()
        reference = str(RAID[11][i]).lower()
        if (Category1 == 'lncrna'and Category2=='protein')or(Category2=='lncrna'and Category1=='protein'):
            for j in range(13):
                f.write(str(RAID[j][i])+"\t")
            f.write("\n")
            if reference != 'none':
                for j in range(13):
                    f1.write(str(RAID[j][i])+"\t")
                f1.write("\n")
        if Category1 =='lncrna'.lower():
            lncRNAs.add(RAID[1][i])
        if Category2 == 'lncrna'.lower():
            lncRNAs.add(RAID[5][i])

        if Category1 == 'protein'.lower():
            proteins.add(RAID[1][i])
        if Category2 == 'protein'.lower():
            proteins.add(RAID[5][i])

    for lncRNA in lncRNAs:
        f_lncRNA.write(str(lncRNA)+"\n")
    for protein in proteins:
        f_protein.write(str(protein)+"\n")
    f.close()
    f1.close()
    f_lncRNA.close()
    f_protein.close()