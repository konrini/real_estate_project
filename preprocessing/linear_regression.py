import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data_linear = pd.read_csv('C:/Users/user/Downloads/Data_linear.csv')
data_polynomial = pd.read_csv('C:/Users/user/Downloads/Data_polynomial.csv')

clf = linear_model.LinearRegression()

x1, y1 = data_linear.iloc[:-20,2:-1], data_linear.iloc[:-20,-1]
x2, y2 = data_polynomial.iloc[:-20,2:-1], data_polynomial.iloc[:-20,-1]

fig = plt.figure(figsize=(8,8)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

clf.fit(x1, y1)
y_pred1 = clf.predict(data_linear.iloc[-20:,2:-1])
ax.plot(data_linear.iloc[-20:,1], y_pred1, label='Linear regression_linear')

clf.fit(x2, y2)
y_pred2 = clf.predict(data_polynomial.iloc[-20:,2:-1])
ax.plot(data_linear.iloc[-20:,1], y_pred2, label='Linear regression_polynomial')

y_true = data_linear.iloc[-20:,-1]
ax.plot(data_linear.iloc[-20:,1], y_true)

ax.legend()
plt.show()