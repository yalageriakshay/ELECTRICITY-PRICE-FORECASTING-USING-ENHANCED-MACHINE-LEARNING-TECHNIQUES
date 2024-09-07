import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

df = pd.read_csv('C:/Users/YMTS0356/Downloads/Datasets/electricity price prediction/energy_dataset.csv')
df.head()
#df.shape
df['price day ahead']
df.isnull().sum()
x = df.drop(['price day ahead','generation hydro pumped storage aggregated','forecast wind offshore eday ahead'],axis = 1)
x.shape
y = df['priceactual']
#x.select_dtypes(include='object')
x.isnull().sum()
x.info()
x.fillna(x.median(),inplace=True)
x.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
a = le.fit_transform(x['time'])
print(a)
x.shape
x.drop(['time'],axis = 1,inplace=True)
x['Time'] = a
x.head()
x.drop(['generation fossil coal-derived gas','generation fossil oil shale','generation fossil peat','generation geothermal','generation marine','generation wind offshore'],axis = 1,inplace=True)
X =x[['generationfossilgas','generationfossilhardcoal','generationhydropumpedstorageconsumption','generationhydrowaterreservoir','generationotherrenewable','generationwaste','totalloadforecast','totalloadactual','Time','priceactual']]
X.to_csv('cleaned_data.csv')
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
model = rfr.fit(x_train,y_train)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
pred = model.predict(x_test)
print(pred)
mean_squared_error(y_test,pred)
model.score(x_test,y_test)
mean_absolute_error(y_test,pred)
r2_score(y_test,pred)
# # from sklearn.svm import SVR
# # svr = SVR()
# # model1 = svr.fit(x_train,y_train)
# # r2_score(y_test,model1.predict(x_test))
# xgbr = xgb.XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=25)
# model2 = xgbr.fit(x_train, y_train)
# pred2 = model2.predict(x_test)
# score = r2_score(y_test, pred2)
# a = model2.feature_importances_
# pd.Series(a)
# # from sklearn.model_selection import GridSearchCV
# parameters = [{'kernel': ['rbf','poly'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2],'C': [1, 10, 100]}]
# c = GridSearchCV(SVR(),param_grid=parameters,cv=5)
# c.fit(x_train,y_train)
# print(c.best_params_)
d = [4,5,6]
e = ['a','b','c']
f ={}
for i in range(len(e)):
    g =e[i]
    f.append(g,d[i])
print(f)