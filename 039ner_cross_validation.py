# encoding = utf-8
import pandas as pd
from sklearn.model_selection import KFold

if __name__ == '__main__':
    file = pd.read_csv("./out/039sample2wordPOS_utf8.txt",header=None,sep="\t",encoding="gbk")
    floder = KFold(n_splits=10, random_state=0, shuffle=False)
    num = 0
    path = "./out/039/"
    for train_loc, test_loc in floder.split(file):
        f = open(path+"039sample2wordPOS_train"+str(num)+".txt","w",encoding="utf-8")
        for i in train_loc:
            f.write(str(file[0][i])+"\t"+str(file[1][i])+"\n")
        f.close()

        f = open(path+"039sample2wordPOS_test"+str(num)+".txt","w", encoding="utf-8")
        for i in test_loc:
            f.write(str(file[0][i]) + "\t" + str(file[1][i]) + "\n")
        f.close()
        num += 1

    print()