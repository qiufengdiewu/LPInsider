#coding:utf-8

import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

#下载数据
train_vec = np.load('./out/009train_vec_wiki.npy')
test_vec = np.load('./out/009test_vec_wiki.npy')
y_train =np.load('./out/007y_train_data.npy')
y_test = np.load('./out/007y_test_data.npy')
#训练SVM模型
def svm_tran(train_vec,y_train,test_vec,y_test):
    clf = SVC(kernel='rbf',verbose=True)
    clf.fit(train_vec,y_train)
    #持久化保存模型
    joblib.dump(clf,'./out/010svm_model_wiki.pkl',compress=3)
    print(clf.score(test_vec,y_test))

svm_tran(train_vec,y_train,test_vec,y_test)