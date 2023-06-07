#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: imports
# @time: 2023/4/6,17:46


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def generate_clfs(random_state=None):  # 由于有类似随机森林的属性扰动，可能不唯一设置个random_state
    names = ['感知器',
             '逻辑斯蒂回归',
             'K-近邻',
             '朴素贝叶斯（先验高斯分布）',
             '线性支持向量机',
             '高斯核的支持向量机',
             '多项式核的支持向量机',
             '决策树',
             '随机森林',
             'Adaboost',
             '梯度提升树']
    classifiers = [
        Perceptron(eta0=0.2),  # 学习率0.2
        LogisticRegression(solver='liblinear'),  # 梯度下降
        KNeighborsClassifier(5),  # 最近的5个
        GaussianNB(),  # 先验高斯的贝叶斯
        SVC(kernel="linear", C=1),  # 线性核
        SVC(gamma=2, C=1),  # 高斯核
        SVC(kernel='poly', C=1, degree=3, gamma='auto', coef0=0),  # 多项式核
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10,max_features=1),
        AdaBoostClassifier(n_estimators=50),
        GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1)
    ]
    if random_state is not None:
        for clf in classifiers:
            clf.random_state = random_state
    return dict(zip(names, classifiers))
