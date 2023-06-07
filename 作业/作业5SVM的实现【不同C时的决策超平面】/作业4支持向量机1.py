#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:GJH 
# time:2023/3/10
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']

# 样本数据
X = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])

# 创建SVM分类器对象
C = [1, 0.2]
clf_list = []

fig, ax = plt.subplots(figsize=(8, 6), dpi=80)

for c in C:
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(X, y)
    clf_list.append(clf)

# 绘制SVM超平面及间隔边界
for clf, c, color, linestyle in zip(clf_list, C, ['red', 'blue'], ['-', '--']):
    # 获取SVM超平面
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0.5, 4)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # 获取SVM间隔边界
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.abs(a) * margin
    yy_up = yy + np.abs(a) * margin

    # 绘制SVM超平面及间隔边界
    ax.plot(xx, yy, label='SVM, C={}'.format(c), color=color, linestyle=linestyle)
    ax.plot(xx, yy_down, color=color, linestyle=linestyle)
    ax.plot(xx, yy_up, color=color, linestyle=linestyle)

    # 输出对偶模型的最优解alpha
    print('C={}时，对偶模型的最优解alpha为：{}'.format(c, clf.dual_coef_))
    # 输出支持向量
    print('C={}时，支持向量为：\n{}'.format(c, clf.support_vectors_))

# 绘制样本数据
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='*', s=250)

# 添加图例
ax.legend(loc='best',fontsize=15)

# 添加坐标轴标签和标题
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('SVM分类',fontsize=25)


plt.show()
