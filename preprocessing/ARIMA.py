#######################  ARIMA  #######################
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import seaborn as sns


file_path = 'C:/Users/user/Downloads/data.csv'
# df = pd.read_csv(file_path, names=['month', 'transaction', 'mart', 'subway', 'interest rate', 'school', 'age', 'CPI', 'resident', 'price'])
df = pd.read_csv(file_path, names=['month', 'mart', 'subway', 'industry', 'interest rate', 'resident', 'price'])

df['month'] = pd.to_datetime(df['month'])

df.index = df['month']
df.set_index('month', inplace=True)

# (AR=8, 차분=1, MA=3) 파라미터로 ARIMA 모델을 학습합니다.
model = ARIMA(df.price.values, order=(8, 1, 3))

# trend : constant를 가지고 있는지, c - constant / nc - no constant
# disp : 수렴 정보를 나타냄
model_fit = model.fit(trend='nc', full_output=True, disp=True)
print(model_fit.summary())

forecast_data = model_fit.forecast(steps=12)  # 학습 데이터셋으로부터 5개월 뒤를 예측합니다.

# 테스트 데이터셋을 불러옵니다.
test_file_path = 'C:/Users/user/Downloads/target.csv'
test_df = pd.read_csv(test_file_path, names=['month', 'y'])

pred_y = forecast_data[0].tolist()  # 마지막 12개월의 예측 데이터입니다.
test_y = test_df.y.values  # 실제 12개월의 가격 데이터입니다.

print('ARIMA 모델 : ', 'AR=8, 차분=1, MA=3')
print('선택한 변수 : ', df.columns)
for i in range(12):
    print('실제 가격 :', test_y[i], ',', '예측 가격 :', pred_y[i])
# pred_y_lower = []  # 마지막 12개월의 예측 데이터의 최소값입니다.
# pred_y_upper = []  # 마지막 12개월의 예측 데이터의 최대값입니다.
# for lower_upper in forecast_data[2]:
#     lower = lower_upper[0]
#     upper = lower_upper[1]
#     pred_y_lower.append(lower)
#     pred_y_upper.append(upper)

plt.plot(pred_y, color="gold", label='predicted')  # 모델이 예상한 가격 그래프입니다.
plt.plot(test_y, color="green", label='expected')  # 실제 가격 그래프입니다.
plt.legend()
# plt.plot(pred_y_lower, color="red")  # 모델이 예상한 최소가격 그래프입니다.
# plt.plot(pred_y_upper, color="blue")  # 모델이 예상한 최대가격 그래프입니다.
plt.show()