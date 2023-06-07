#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: Main_comparsion
# @time: 2023/4/6,20:15
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

import Appointed_clfs
import Visual_funs


def load_data():
    datasets = {}
    # 路径简单写了，放在和文件同一个路径
    for path in ['data_circles.csv',
                 'linearly_separable.csv',
                 'data_moons.csv']:
        datasets[path.split('.')[0]]=Visual_funs.pd.read_csv(path, header=None)
    return datasets


if __name__ == '__main__':
    # 初始化
    clf_dict = Appointed_clfs.generate_clfs(2023)  # 所有的算法，字典（名称，算法）
    datasets = load_data()  # 加载数据
    plot_ax = partial(Visual_funs.create_ax, test_size=0.3)  # 固定参数，使得不必在循环里重传，降低复杂度【使用默认值的设置不能在其他入口调整】

    # 循环作图
    fig, ax = plt.subplots(len(datasets), len(clf_dict)+1, figsize=(33, 9), dpi=500)#分辨率，和决策边界画的点比较多，画图比较慢

    for row, (df_name,df) in enumerate(datasets.items(), 0):
        X, y = df.iloc[:, :2], df.iloc[:, 2]  # 取出X，y
        plot_ax(X,y,ax[row, 0],clf=None, clf_name=df_name)#只画点，不画决策边界
        for col, (clf_name, clf) in enumerate(tqdm(clf_dict.items()),1):
            plot_ax(X,  # 特征
                    y,  # 类别
                    ax[row, col],  # 当前子图
                    clf,  # 分类器
                    clf_name,  # 分类器显示的中文
                    )
    plt.tight_layout()
    plt.show()
    fig.savefig('Comparing.png')

    # ---------------------------------------------------------------------------------------备注
    # fig = plt.figure(figsize=(x,x))  # 首先调用plt.figure()创建了一个**画窗对象fig**
    # ax = fig.add_subplot(111)  # 然后再对fix创建默认的坐标区（一行一列一个坐标区）
    # # 这里的（111）相当于（1，1，1），当然，官方有规定，当子区域不超过9个的时候，我们可以这样简写
