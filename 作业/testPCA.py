# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:32:39 2018

@author: dx.z
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
X = pd.read_csv('dataForPCA.csv', header=None)
X= pd.DataFrame([[0,1,0,2,3,2],[0,0,1,2,2,3]]).T
pca = PCA(n_components=2,
          svd_solver = 'auto')
pca.fit(X)
# 训练样本均值，用于中心化过程
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

m = pca.mean_
print('平均值:\n{}'.format(pca.mean_))
print('-'*40)

# 协方差矩阵的特征根：在对应维度上投影后的方差
print('特征根:\n{}'.format(pca.explained_variance_))
print('-'*40)

# 特征根占的比例
r = pca.explained_variance_ratio_
print('每个特征根占比:\n{}'.format(r))
print('-'*40)
# 累积贡献率
print('累积贡献率:\n{}'.format(r.sum()))
print('-'*40)
# 用于降维的投影矩阵
M = pca.components_
print('投影矩阵:\n{}'.format(M))
print('-'*40)
# 对第一个样本降维
y = M.dot(X.iloc[0,:]-m)
print('第一个样本为:\n{}'.format(X.iloc[0,:].to_numpy()))
print('使用投影矩阵对第一个样本降维:\n{}'.format(y))
y_ = pca.transform(X.iloc[0,:].to_numpy().reshape(1, -1))
print('使用成员方法对第一个样本降维:\n{}'.format(y_))

print('-'*40)
# 恢复第一个样本
x_ = y.dot(M) + m
print('使用投影矩阵恢复第一个样本:\n{}'.format(x_))
x__ = pca.inverse_transform(y_)
print('使用成员方法恢复第一个样本:\n{}'.format(x__))