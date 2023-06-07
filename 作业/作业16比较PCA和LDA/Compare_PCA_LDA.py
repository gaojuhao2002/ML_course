#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: Compare_PCA_LDA
# @time: 2023/5/9,16:20
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

np.random.seed(1)

# 加载数据集
digits = load_digits()
X = digits.data
y = digits.target

# 随机选择3个数字类别
class_indices = np.random.choice(np.unique(y), size=3, replace=False)
X_selected = X[np.isin(y, class_indices)]
y_selected = y[np.isin(y, class_indices)]


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_selected)
# 使用LDA降维到2维
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_selected, y_selected)


# 可视化PCA降维后的数据
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i in range(3):
    label=class_indices[i]
    c=colors[i]
    # 随机选择20个样本
    P_X1=X_pca[y_selected == label, 0]
    P_X2=X_pca[y_selected == label, 1]
    samples_indices = np.random.choice([x for x in range(len(P_X1))], size=20, replace=False)
    P_X1,P_X2=P_X1[samples_indices],P_X2[samples_indices]
    plt.scatter(P_X1,P_X2, color=c, alpha=0.8, label=label)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of selected digits (n_components = 2)')
plt.show()

# 可视化LDA降维后的数据
plt.figure(figsize=(8, 6))
for i in range(3):
    label=class_indices[i]
    c=colors[i]
    # 随机选择20个样本
    P_X1=X_lda[y_selected == label, 0]
    P_X2=X_lda[y_selected == label, 1]
    samples_indices = np.random.choice([x for x in range(len(P_X1))], size=20, replace=False)
    P_X1,P_X2=P_X1[samples_indices],P_X2[samples_indices]
    plt.scatter(P_X1,P_X2, color=c, alpha=0.8, label=label)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of selected digits (n_components = 2)')
plt.show()
