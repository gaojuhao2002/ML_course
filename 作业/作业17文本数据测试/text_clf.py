from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB

from sklearn.metrics import f1_score
def train_val(D,vect,nb,to_array=False):
    X_train,X_test, y_train, y_test=D

    X_train_vect=vect.fit_transform(X_train)
    X_test_vect=vect.transform(X_test)
    if to_array:
        X_train_vect,X_test_vect=X_train_vect.toarray(),X_test_vect.toarray()
    nb.fit(X_train_vect,y_train)
    pred=nb.predict(X_test_vect)
    f1_weighted=f1_score(y_test,pred,average='macro')
    # df_X_train=pd.DataFrame(X_train_ved.toarray(),columns=vectorizer.get_feature_names_out()) #查看矩阵形式
    return f1_weighted
#--------------实验1：应该使用GaussianNB还是MultinomialNB？
#由于内存问题，对比时选用四类数据
news = fetch_20newsgroups(subset='all',categories=['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space'])
D =train_test_split(news.data,news.target,test_size=0.2, random_state=2023)
#控制变量，使用相同的特征提取器
vect=TfidfVectorizer(stop_words='english')
#使用两种分类器
mnb=MultinomialNB(alpha=0.1)
gnb=GaussianNB()
#预测且评价
print('MultinomialNB的加权F1',train_val(D,vect,mnb))
print('GaussianNB的加权F1',train_val(D,vect,gnb,True))

# 实验1结论：性能上，MultinomialNB表现更优，且接收稀疏矩阵传入，大大减少了时空复杂度，因此该选用MultinomialNB。对于后续实验，选定分类器为MultinomialNB，此时由于其接收稀疏矩阵的特性，可以选取所有的数据集，而不用固定4类
#-------------------------实验2：使用相同的训练集和测试集，比较CountVectorizer和TfidfVectorizer的效果¶
#-------------------------实验3：考察停用词的作用
news = fetch_20newsgroups(subset='all')
D =train_test_split(news.data,news.target,test_size=0.2, random_state=2023)
#控制变量，使用相同的分类器
mnb=MultinomialNB()
#使用两种特征提取器
cv_sw=CountVectorizer(stop_words='english')
tv_sw=TfidfVectorizer(stop_words='english')
cv=CountVectorizer()
tv=TfidfVectorizer()
#预测且评价
print('CountVectorizer的加权F1(无停用词)',train_val(D,cv,mnb))
print('TfidfVectorizer的加权F1（无停用词）',train_val(D,tv,mnb))
print('CountVectorizer的加权F1(有停用词)',train_val(D,cv_sw,mnb))
print('TfidfVectorizer的加权F1（有停用词）',train_val(D,tv_sw,mnb))
#-------------------------结论：【回答问题二】一方面，在同有停用词或同无停用词的条件下，CountVectorizer与TfidfVectorizer的性能差异不大。【回答问题 三】另一方面，无论CountVectorizer或TfidfVectorizer都被停用词的引入显著提高了预测效果
#----------------------------实验 4：考察Tf–idf 平滑的作用
tv=TfidfVectorizer()
tv_no_smooth=TfidfVectorizer(smooth_idf=False)
print('TfidfVectorizer的加权F1（有平滑）',train_val(D,tv,mnb))
print('TfidfVectorizer的加权F1（无平滑）',train_val(D,tv_no_smooth,mnb))
#---------------------------------结论：有平滑小小的提升
