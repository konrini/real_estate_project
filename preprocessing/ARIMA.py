#######################  ARIMA  #######################

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import font_manager, rc
import matplotlib
import statsmodels.graphics.tsaplots as sgt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


font_path = "C:/Windows/Fonts/malgunbd.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False


file_path = 'C:/Users/user/Downloads/data2.csv'
df = pd.read_csv(file_path, names=['month', 'shop', 'subway', 'interest rate', 'resident', 'transaction', 'school', 'age', 'CPI', 'price'])
# print(df)
df['month'] = pd.to_datetime(df['month'])

df.index = df['month']
df.set_index('month', inplace=True)

Scaler = MinMaxScaler()
Scaled_X = Scaler.fit_transform(df.iloc[:,1:-1])
df.iloc[:,1:-1] = Scaled_X

plt.plot(df['price'])
plt.title('분당구 평당가')
plt.show()

# df['price'] = np.log1p(df['price'])
# plt.plot(df['price'])
# plt.title('로그 변환한 분당구 평당가')
# plt.show()
#
# diff_1 = df['price'].diff()
# df['price'] = diff_1
# df = df.dropna()
# plt.plot(df['price'])
# plt.title('1차 차분')
# plt.show()

result = seasonal_decompose(df['price'], model='additive', freq=12)
result.plot()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
sgt.plot_acf(df['price'], lags = 20, zero = False, ax=ax1)
ax1.set_title("ACF")
sgt.plot_pacf(df['price'], lags = 20, zero = False, method = ('ols'), ax=ax2)
ax2.set_title("PACF")
plt.show()

# ADF는 추세가 제거 되었는지 확인하는데 유용하고, KPSS는 계절성 제거가 되었는지 확인하는데 유용하다.
result = adfuller(df['price'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
# p-value가 0.05를 넘으므로 귀무가설을 기각하지 못한다. 즉, 해당 데이터는 정상성을 만족하지 못한다.

# 1차 차분
# diff_1 = df['price'].diff()
# df['price'] = diff_1
# df = df.dropna()
# plt.plot(df['price'])
# plt.title('1차 차분')
# plt.show()
#
# result = adfuller(df['price'])
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
#     print('\t%s: %.3f' % (key, value))
# # p-value가 0.05를 넘지 않으므로 귀무가설을 기각한다. 즉, 해당 데이터는 정상성을 만족한다.
#
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
# sgt.plot_acf(df['price'], lags = 20, zero = False, ax=ax1)
# ax1.set_title("ACF")
# sgt.plot_pacf(df['price'], lags = 20, zero = False, method = ('ols'), ax=ax2)
# ax2.set_title("PACF")
# plt.show()

# 2차 차분
# diff_2 = diff_1.diff().dropna()
# plt.plot(diff_2)
# plt.title('2차 차분')
# plt.show()

stepwise_fit = auto_arima(df['price'], trace=True, suppress_warnings=True)

# (AR=8, 차분=1, MA=3) 파라미터로 ARIMA 모델을 학습합니다.
model = ARIMA(df.price.values, order=(1, 2, 1))

# trend : constant를 가지고 있는지, c - constant / nc - no constant
# disp : 수렴 정보를 나타냄
model_fit = model.fit(trend='nc', full_output=True, disp=True)
print(model_fit.summary())

forecast_data = model_fit.forecast(steps=12)  # 학습 데이터셋으로부터 12개월 뒤를 예측합니다.

# 테스트 데이터셋을 불러옵니다.
test_file_path = 'C:/Users/user/Downloads/target2.csv'
test_df = pd.read_csv(test_file_path, names=['month', 'shop', 'subway', 'interest rate', 'resident', 'transaction', 'school', 'age', 'CPI', 'y'])

pred_y = forecast_data[0].tolist()  # 마지막 12개월의 예측 데이터입니다.
test_y = test_df.y.values  # 실제 12개월의 가격 데이터입니다.

print('ARIMA 모델 : ', 'AR=1, 차분=2, MA=1')
print('선택한 변수 : ', df.columns)
for i in range(12):
    print('실제 가격 :', test_y[i], ',', '예측 가격 :', pred_y[i])
# pred_y_lower = []  # 마지막 12개월의 예측 데이터의 최소값입니다.
# pred_y_upper = []  # 마지막 512월의 예측 데이터의 최대값입니다.
# for lower_upper in forecast_data[2]:
#     lower = lower_upper[0]
#     upper = lower_upper[1]
#     pred_y_lower.append(lower)
#     pred_y_upper.append(upper)

plt.plot(pred_y, color="gold", label='predicted')  # 모델이 예상한 가격 그래프입니다.
plt.plot(test_y, color="green", label='expected')  # 실제 가격 그래프입니다.
# plt.plot(pred_y_lower, color="red")  # 모델이 예상한 최소가격 그래프입니다.
# plt.plot(pred_y_upper, color="blue")  # 모델이 예상한 최대가격 그래프입니다.
plt.xticks(ticks=range(0, 12), labels=range(1, 13))
plt.xlabel('2020년(월)')
plt.ylabel('평당가(만원)')
plt.title('ARIMA')
plt.legend()
plt.show()

rmse=sqrt(mean_squared_error(pred_y, test_df['y']))
print(rmse)