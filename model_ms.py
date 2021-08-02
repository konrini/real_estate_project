import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf


df = pd.read_csv('판교푸르지오그랑블실거래가_110713-210713.csv', encoding = 'utf-8')

# data 전처리
from datetime import datetime
df['계약날짜'] = df['계약년월']
for i in range(len(df['거래금액(만원)'])):
    df['계약날짜'][i] = str(df['계약년월'][i]) + str(df['계약일'][i]).zfill(2)
    df['계약날짜'][i] = datetime.strptime(str(df['계약날짜'][i]), "%Y%m%d").date()
    df['거래금액(만원)'][i] = int(''.join(df['거래금액(만원)'][i].split(',')))

df['평당가격'] = df['거래금액(만원)']
for i in range(len(df['거래금액(만원)'])):
    df['평당가격'][i] = int((df['거래금액(만원)'][i]/df['전용면적(㎡)'][i])*3.3)

df['월평균가격'] = df.groupby('계약년월').transform(np.mean)['평당가격'].astype(int)

# 변화율 >> 감소: -1, 유지: 0, 증가: 1
apart_mean_price = df[['시군구', '계약년월', '계약날짜', '평당가격', '월평균가격']]
apart_mean_price['변화율'] = apart_mean_price['평당가격']
apart_mean_price['변화율'][0] = 0
for i in range(1, len(apart_mean_price['평당가격'])):
    if apart_mean_price['평당가격'][i] - apart_mean_price['평당가격'][i - 1] < 0:
        apart_mean_price['변화율'][i] = -1

    elif apart_mean_price['평당가격'][i] - apart_mean_price['평당가격'][i - 1] == 0:
        apart_mean_price['변화율'][i] = 0

    else:
        apart_mean_price['변화율'][i] = 1

# 개통일 구간별로 label 변경
bins = [0, 201110, 201601, 201609, 202200]
group_n = ['none', '1st', '2nd', 'gg']
open_cuts = pd.cut(apart_mean_price['계약년월'], bins, labels = group_n)

# dummies
open_dum = pd.get_dummies(open_cuts)
open_dum = open_dum.add_prefix('open_')

# 생성된 dummies 랑 data 하나로 합치기
total_apart = pd.concat([apart_mean_price, open_dum], axis = 1)

# data 정리 >> 평당가격: label
#             개통일별 dummy, 변화율: 변수
df = total_apart[['평당가격','open_none', 'open_1st', 'open_2nd', 'open_gg','변화율']]
df['평당가격']=df['평당가격'].astype(int)
df['변화율'] = df['변화율'].astype(int)

# modeling >> seed 값 지정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

# modeling >> dataframe 형태에서 data 값들만 추출
dataset = df.values
X = dataset[:,1:]
Y = dataset[:,0]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.3, random_state = seed)

model = Sequential()
model.add(Dense(30, input_dim = 5, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(X_train, Y_train, epochs = 1000, batch_size = 10)

Y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print('실제가격: {:.3f}, 예상가격: {:.3f}'.format(label, prediction))

# # 그래프 폰트 설정
# font_path = "C:/Windows/Fonts/malgun.ttf"
# font_name = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font_name)
#
# plt.plot(df['계약년월'], df['평당가격'], 'co')
# plt.title('월별 평균 평당가격')
# plt.xlabel('계약년월')
# plt.ylabel('평당가격')
# plt.show()
#
# plt.plot(test['계약날짜'], test['평당가격'], 'co')
# plt.title('월별 평균 평당가격')
# plt.xlabel('계약날짜')
# plt.ylabel('평당가격')
# plt.show()