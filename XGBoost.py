import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

table = pd.read_csv('C:/Users/user/Desktop/교육/프로젝트/최종/데이터/최종 데이터/prac_data_mart.csv', encoding = 'utf-8')
table['new_date'] = pd.to_datetime(table['date'], format='%Y-%m-%d')
print(table.head())
print("\n")
print(table.info())

table['y-m'] = table['new_date'].dt.strftime('%Y-%m')
table.drop(['date', 'new_date'], axis = 1, inplace = True)
table.set_index('y-m', inplace = True)

print(table.head())
print('\n')
print(table.info())


# 데이터 변환(DMatrix 형태)
import xgboost as xgb
from sklearn.metrics import mean_squared_error

X, y = table.iloc[:,:-1], table.iloc[:,-1]
table_dmatrix = xgb.DMatrix(data = X, label = y)


# 모델 학습
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = False, test_size = 0.09, random_state = 123)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

xg_reg = xgb.XGBRegressor(objective = 'reg:linear', colsample_bytree = 0.2, learning_rate = 0.1, max_depth = 2, alpha = 2, n_estimators = 10)

xg_reg.fit(X_train, y_train)


# 예측
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# Cross Validation
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=table_dmatrix, params=params, nfold=3,num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

cv_results.head()

print((cv_results["test-rmse-mean"]).tail(1))

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()