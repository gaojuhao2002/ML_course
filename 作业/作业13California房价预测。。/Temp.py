#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: Temp
# @time: 2023/4/20,15:25
#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: California预测
# @time: 2023/4/15,14:28
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import HuberRegressor,LinearRegression,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor



def generate_rgs(ss=True,random_state=2023):
    names = ['LinearRegression',
             'Ridge',
             'HuberRegressor',
             'KNeighborsRegressor',
#              'mobel_svr_rbf',
#              'mobel_svr_poly',
#              'mobel_svr_linear',
             'DecisionTreeRegressor',
             'GradientBoostingRegressor',
             'RandomForestRegressor'
             ]
    regressioners = [
        LinearRegression(copy_X=True,fit_intercept=True),
        Ridge(alpha=1.5),
        HuberRegressor(alpha=1,epsilon=1.35),
        KNeighborsRegressor(n_neighbors=5),
#         SVR(kernel='rbf'),
#         SVR(kernel='poly'),
#         SVR(kernel='linear'),
        DecisionTreeRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor()
    ]


    if ss:
        regressioners=[make_pipeline(StandardScaler(), rgs) for rgs in regressioners]

    if random_state is not None:
        for rgs in regressioners:
            rgs.random_state = random_state
    return dict(zip(names, regressioners))


def cal_(y_pre, y_test):
    mse = mean_squared_error(y_pre,y_test)
    mae = mean_absolute_error(y_pre,y_test)
    r2 = r2_score(y_pre,y_test)
    return mse,mae,r2
def trainer(data,target,rad=2023,ss=False):
    X_train,X_test,y_train,y_test = train_test_split(data,target,test_size = 0.2,random_state = rad)
    rgs_dict=generate_rgs(ss,rad)
    for (name,rgs) in (rgs_dict.items()):
        rgs.fit(X_train, y_train)
        y_pre=rgs.predict(X_test)
        mse,mae,r2=cal_( y_test,y_pre)
        print(name,':\t','mse={}\tmae={}\tr2={}\t'.format(mse,mae,r2))
if __name__=='__main__':
    california_housing = fetch_california_housing()

    data = california_housing.data

    target = california_housing.target
    trainer(data, target, rad=2023)
    trainer(data, target, rad=2023, ss=True)
