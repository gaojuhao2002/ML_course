#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: test
# @time: 2023/4/24,22:58
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

x = np.array([[0,0],[2,1],[2,-1],[2.1,0],[4.1,0],[5,0]])
data = pd.DataFrame(x,columns=['x1','x2'])

inits = np.array([[[2.1,0],[4.1,0],[5,0]],
                 [[0,0],[2.1,0],[4.1,0]],
                 [[0,0],[4.1,0],[5,0]]])  # 自定义初始聚类中心
j=1
for i in range(inits.shape[0]):
    kmeans = KMeans(n_clusters = 3,   # 聚成3类
                    init = inits[i],  # 自定义聚类中心
                    random_state = 10)
    kmeans.fit(x)
    centers = kmeans.cluster_centers_
    print('聚类中心：\n',np.round(centers,2))  # 聚类中心
    print('迭代次数：{}'.format(kmeans.n_iter_))
    labels = kmeans.labels_
    plt.figure(j)
    one = data[labels == 0]
    two = data[labels == 1]
    three = data[labels == 2]
    plt.scatter(one['x1'],one['x2'],alpha=0.8,s=200,c='#96577F')             # 显示据类结果
    plt.scatter(two['x1'],two['x2'],alpha=0.8,s=200,c='#FABE44')
    plt.scatter(three['x1'],three['x2'],alpha=0.8,s=200,c='#1F3095')
    plt.scatter(centers[:,0],centers[:,1],alpha=0.8,s=150,c='r',marker='*')  # 显示聚类中心
    plt.show()
    print('\n')
    j+=1
