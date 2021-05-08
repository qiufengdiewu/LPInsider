# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 生成画布
plt.figure(figsize=(20, 8), dpi=80)
# 横坐标名字
#movie_name = ['雷神3：诸神黄昏', '正义联盟', '东方快车谋杀案', '寻梦环游记', '全球风暴', '降魔传', '追捕', '七十七天', '密战', '狂兽', '其它']
x_name = ['SVM','逻辑回归','随机森林','xgboost','lightGMB']
# 纵坐标数量
#y1 = [73853, 57767, 22354, 15969, 14839, 8725, 8716, 8318, 7916, 6764, 52222]
#y2 = [12345, 54321, 23456, 34567, 45678, 56789,312,  123,  333,  444,  6666]
y_num= [0.91220679,0.904753086, 0.844197531,0.887469136,0.89242284]

#x = np.arange(len(movie_name))
x_np = np.arange(len(x_name))

bar_width=0.3

#plt.bar(x, y1, width=bar_width, color=['b', 'r', 'g', 'y', 'c', 'm', 'y', 'k', 'c', 'g', 'g'])
plt.bar(x_name,y_num,width=bar_width,color=['b','r','g','y','c'])
#plt.bar(x+bar_width,y2,width=bar_width)

#plt.xticks(x, movie_name)
plt.xticks(x_np,x_name)
plt.show()

