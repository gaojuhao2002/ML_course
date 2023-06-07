#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2023/3/10
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 读取数据
data = np.loadtxt('excer_svm.csv', delimiter=',', skiprows=1)

# 特征和标签
X = data[:, :2]
y = data[:, 2]

# 创建SVM分类器对象
clf = svm.SVC(kernel='rbf', gamma=0.5, C=1)
clf.fit(X, y)

# 可视化样本空间的分类结果
# 创建网格数据
xx, yy = np.meshgrid(np.linspace(-3, 4, 500), np.linspace(-3, 4, 500))
# 预测网格数据的标签
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制样本数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

# 绘制分类边界
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)

# 绘制支持向量
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', edgecolors='k')


# 绘制待分类的点
test_points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
test_results = clf.predict(test_points)
plt.scatter(test_points[:, 0], test_points[:, 1], c=test_results, cmap=plt.cm.Paired, marker='*', s=200, linewidths=3)

# 输出预测结果
print('待分类的点的分类结果为：', test_results)
plt.show()
