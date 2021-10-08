import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import datetime as datetime
import matplotlib.pyplot as plt
'''
 column명 : ['Transaction_qt', 'Mart_cnt', 'Subway_cnt', 'Interest_rate', 'School_cnt', 'Apt_age', 'CPI', 'total_population','bundang_population', 'population_rate', 'Cost']
 LSTM 다변량 예측 모델
'''
df = pd.read_csv('모델링 마트 테이블_분당구.csv', parse_dates=['Date'], index_col=['Date'], infer_datetime_format=True, encoding='utf-8',)
df = df[['Mart_cnt', 'Interest_rate', 'CPI', 'Cost']]
print(df.head())

values = df.values
print(values)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print(scaled.shape)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = pd.DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
      cols.append(df.shift(i))
      names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
      cols.append(df.shift(-i))
      if i == 0:
          names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
      else:
          names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
      agg.dropna(inplace=True)
  return agg

reframed = series_to_supervised(scaled, 1, 1)
print(reframed.columns)
reframed.drop(reframed.columns[[4,5,6]], axis=1, inplace=True)
print(reframed.columns)

values = reframed.values
n_train_hours = 12 * 10 #10년치 데이터만 가져온다
train = values[:n_train_hours, :]
val = values[n_train_hours:n_train_hours + 12, :]
test =values[n_train_hours + 12:, :]
print(train)
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
val_X, val_y = val[:, :-1], val[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences= True))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = MODEL_DIR + "{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                                   verbose=1, save_best_only=True)
esc = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_X, train_y, epochs=200, batch_size=72, callbacks = [esc, checkpointer],
        validation_data=(val_X, val_y), verbose=1, shuffle=False)
print(model.summary())
score = model.evaluate(test_X, test_y, batch_size=72)
print('val_mse: {0:0.4f}'.format(score))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('train-validation loss')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
# print(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('test_rmse: {0:0.4f}'.format(rmse))
# print(inv_yhat)
# print(inv_y)
plt.plot(df['2021-02':].index, inv_yhat, label = 'Predict')
plt.plot(df['2021-02':].index, inv_y, label = 'Actual')
plt.title('2021-02~2021-07 LSTM model')
plt.legend()
plt.show()

