import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

digits = pd.read_csv('C:/Users/user/Downloads/Data_linear.csv')

clf = linear_model.LinearRegression()

x, y = digits.iloc[:-30,2:-1], digits.iloc[:-30,-1]

clf.fit(x, y)

y_pred = clf.predict(digits.iloc[-30:,2:-1])
y_true = digits.iloc[-30:,-1]

fig = plt.figure(figsize=(8,8)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

ax.plot(digits.iloc[-30:,1], y_pred)
ax.plot(digits.iloc[-30:,1], y_true)
plt.show()