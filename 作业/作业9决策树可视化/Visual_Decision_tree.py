#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: Visual_Decision_tree
# @time: 2023/3/23,21:02
#------------------------------------------------------read_data
import pandas as pd
from sklearn import tree

class Named_tree():
    def __init__(self,dct,type_,criterion):
        self.dct=dct
        self.type_=type_
        self.criterion=criterion

#______________________________________________________________encode
from sklearn.preprocessing import LabelEncoder
def en_de_code_context(data,unco_context_cols):
    les={}
    map_info=''
    for col in unco_context_cols:
        label=LabelEncoder()
        data[col]=label.fit_transform(data[col])
        # print('encoded',col)
        map_info+=col+':  '+str(dict(zip(label.classes_, label.transform(label.classes_))))+'\n'
        les[col]=label
    return data,les,map_info
def pre_data(df,type_=1):
    need_enc_data=df.copy() if type_==1 else df.copy().iloc[:, [-1]]
    data,les,map_info=en_de_code_context(need_enc_data,need_enc_data.columns.to_list())
    return data if type_==1 else pd.concat([pd.get_dummies(df.iloc[:, :-1]), data], axis=1),les,map_info
def build_tree_visual(X,y,le_y,criterion='gini',type_=1,saving_file_name=None,map_info=None):
    import matplotlib.pyplot as plt
    dct=tree.DecisionTreeClassifier(criterion=criterion)
    dct.fit(X,y)
    plt.figure(figsize=(15,20),dpi=100)
    rule_text=tree.plot_tree(dct,
                   feature_names=X.columns,
                   class_names=le_y.inverse_transform(dct.classes_),
                   filled=True)
    plt.title('Solutions_{}({})'.format(type_,criterion),fontdict={'size': 50})
    if map_info is not None:
        plt.text(-0.01, 0.01,
                map_info,
                fontsize=20,
                verticalalignment='top',
                bbox=dict(boxstyle="round", fc='#e9f5e6', ec="0.5", alpha=0.9))
    if saving_file_name is not None:
        plt.savefig(saving_file_name)
    plt.show()
    return rule_text,plt,dct
def get_axs(df):
    axs=[]
    named_tree_list=[]
    for criterion in ['gini','entropy']:
        for type_ in [1,2]:
            data,les,map_info=pre_data(df,type_)
            _,ax,dct=build_tree_visual(data.iloc[:,:-1],
                                   data.iloc[:,-1],
                                   les['PlayTennis'],
                                   criterion,type_,
                                   'Solutions_{}({}).png'.format(type_, criterion),
                                   map_info)
            named_tree_list.append(Named_tree(dct,type_,criterion))
            axs.append(ax)
    return axs,named_tree_list
def generate_all_samples(type_):
    # 以参考样本空间具有的取值，生成所有的笛卡尔积
    import itertools
    val=itertools.product(['Overcast','Sunny','Rain'],['Hot','Mild','Cool'],
                            ['High','Normal'],['Weak','Strong'])
    val=pd.DataFrame(val,columns=['Outlook','Temperature','Humidity','Wind'])
    return pd.get_dummies(val)
if __name__ == '__main__':
    df = pd.read_excel('PlayTennis训练样本_.xlsx').iloc[:, 1:]
    axs,named_tree_list=get_axs(df)#要求df前面的列是离散，最后一列是类别
    all_samples=generate_all_samples()
    predict_res=pd.DataFrame()
    for named_tree in named_tree_list:
        dct=named_tree.dct
        type_=named_tree.type_
        criterion=named_tree.criterion
        predict_res['Solutions_{}({})'.format(type_,criterion)]=dct.predict(all_samples)
    predict_res