#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: Visual_Decision_tree
# @time: 2023/3/23,21:02
# ------------------------------------------------------read_data
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

def en_de_code_context(data, unco_context_cols):
    les = {}
    map_info = ''
    for col in unco_context_cols:
        label = LabelEncoder()
        data[col] = label.fit_transform(data[col])
        # print('encoded',col)
        map_info += col + ':  ' + str(dict(zip(label.classes_, label.transform(label.classes_)))) + '\n'
        les[col] = label
    return data, les, map_info

def pre_data(df, type_):
    need_enc_data = df.copy() if type_ == 1 else df.copy().iloc[:, [-1]]
    data, les, map_info = en_de_code_context(need_enc_data, need_enc_data.columns.to_list())
    return data if type_ == 1 else pd.concat([pd.get_dummies(df.iloc[:, :-1]), data], axis=1), les, map_info


def generate_all_samples():
    # 以参考样本空间具有的取值，生成所有的笛卡尔积
    import itertools
    val = itertools.product(['Overcast', 'Sunny', 'Rain'], ['Hot', 'Mild', 'Cool'],
                            ['High', 'Normal'], ['Weak', 'Strong'])

    return  pd.DataFrame(val, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])

class Tree_plus():
    def __init__(self, df, type_, criterion):
        self.type_ = type_
        self.criterion = criterion
        self.name='Solutions_{}({})'.format(self.type_, self.criterion)
        self.dct=tree.DecisionTreeClassifier(criterion=criterion)
        self.fit_tree(df)

    def fit_tree(self,df):
        data,self.les,self.map_info=pre_data(df,self.type_)
        self.X,self.y=data.iloc[:,:-1],data.iloc[:,-1]
        self.dct.fit(self.X,self.y)
    def map_X(self,X):
        # assert X.columns.to_list == list(self.les.keys()).remove(self.y.name),'编码器数量与输入维度不符'
        if self.type_==1:
            for col in X.columns:
                X[col]=self.les[col].transform(X[col])
        if self.type_ ==2:
            X=pd.get_dummies(X)
        return X
    def visual_ax(self,Saving=True):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 20), dpi=100)
        le_y=self.les[self.y.name]
        self.rule_text = tree.plot_tree(self.dct,
                                   feature_names=self.X.columns,
                                   class_names=le_y.inverse_transform(self.dct.classes_),
                                   filled=True)

        plt.title(self.name, fontdict={'size': 50})
        if self.map_info is not None:
            plt.text(-0.01, 0.01,
                     self.map_info,
                     fontsize=20,
                     verticalalignment='top',
                     bbox=dict(boxstyle="round", fc='#e9f5e6', ec="0.5", alpha=0.9))
        plt.savefig(self.name+'.png') if Saving else None
        plt.show()
        self.ax=plt


if __name__ == '__main__':
    df = pd.read_excel('PlayTennis训练样本_.xlsx').iloc[:, 1:]
    plus_tree_dict={}
    #批量初始化 树
    for criterion in ['gini','entropy']:
        for type_ in [1,2]:
            tp=Tree_plus(df,type_,criterion)
            plus_tree_dict[tp.name]=tp
    # 批量 绘图并保存到本地【】
    for k,v in plus_tree_dict.items():
        v.visual_ax()
    # 为所有的样本空间预测
    all_samples=generate_all_samples()
    Val_all_samples=pd.DataFrame()

    for k,v in plus_tree_dict.items():
        Val_all_samples[v.name]=v.dct.predict(v.map_X(all_samples.copy()))    # 预测
        Val_all_samples[v.name]=v.les[v.y.name].inverse_transform(Val_all_samples[v.name])    #解码
    # 写出
    pd.concat([all_samples,Val_all_samples],axis=1).to_excel('res.xlsx')
