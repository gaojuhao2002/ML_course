#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: Visual_funs
# @time: 2023/4/6,19:49
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def displayDecisionBoundary(ax, clf, test_score, x1_min, x1_max, x2_min, x2_max):
    # 创建网格数据
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 1000), np.linspace(x2_min, x2_max, 1000))
    # 预测网格数据的标签
    # 用概率来映射，越接近蓝色，越可能是0，接近红色可能是1
    # 没有的就用predict即可
    if (hasattr(clf, 'predict_proba')):
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    # 绘制分类边界
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
    # 显示得分，一般是准确率
    ax.text(
        x1_max - 0.3,
        x2_min + 0.3,
        ("%.2f" % test_score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    # return plt


def create_ax(X,  # 特征
              y,  # 类别
              ax,  # 传入图
              clf,  # 分类器
              clf_name,  # 分类器显示的中文
              test_size=0.3,  # 采样比例
              ss=False #默认不会标准化
              ):
    # 初始化
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    x1_min, x1_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    x2_min, x2_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5

    # 绘制训练数据
    ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=cm_bright, marker='.', edgecolors="k")
    # 绘制测试数据
    ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap=cm_bright, marker='*', alpha=0.4, edgecolors="k")

    # 美化设置
    plt.tight_layout()  # 布局
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(clf_name)

    if clf is not None:
        # 拟合计算，验证集得分
        clf = make_pipeline(StandardScaler(), clf) if ss else clf
        clf.fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        # 绘制决策边界
        displayDecisionBoundary(ax, clf, test_score, x1_min, x1_max, x2_min, x2_max)
    # return ax


if __name__ == '__main__':
    df = pd.read_csv('data_circles.csv', header=None)
    X, y = df.iloc[:, :2], df.iloc[:, 2]

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier()
    fig, ax = plt.subplots(2, 3)  # 两行三列

    # 示例 在第一行的第二三列画图，但是其他不画就是白的
    for i in range(1, 3):
        create_ax(X,  # 特征
                  y,  # 类别
                  ax[0, i],  # 子图
                  clf,  # 分类器
                  '随机森林',
                  )
    # plt.tight_layout()
    plt.show()
