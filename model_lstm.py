import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# name = '모델링 마트 테이블1.csv'
# table = pd.read_csv(name, encoding = 'utf-8')
# table = table[['date','Subway_count','Interest_rate','Cost']]
# # table = table[['date','Mart_count','Subway_count','Interest_rate','Industry_count','Resident_count','Cost']]
#
# for i in range(len(table['date'])):
#     table['date'][i] = ''.join(table['date'][i].split('-'))
#     table['date'][i] = datetime.strptime(str(table['date'][i]), "%Y%m").date()
#
# all_data = table.interpolate(method = 'polynomial', order = 3)
# all_data.set_index('date', inplace = True)
# all_data = all_data.dropna()
# index = pd.DatetimeIndex(all_data.index.values)
# all_data = pd.DataFrame(all_data.values,index,all_data.columns)
all_data = pd.read_csv('모델링 마트 테이블_분당구.csv', parse_dates=['Date'], index_col=['Date'], infer_datetime_format=True, encoding='utf-8')


def ts_train_test_normalize(all_data, time_steps, for_periods):
    """
    input:
        data: dataframe with dates and price data
    output:
        X_train, y_train: data from 2013/1/1-2018/12/31
        X_test : data from 2019-
        sc :     insantiated MinMaxScaler object fit to the training data
    """
    # time_steps = 6
    # for_periods = 1

    # create training and test set
    ts_train = all_data[:'2019'].values
    ts_test = all_data['2020':].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)
    feature_counts = len(all_data.columns)

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    for i in range(time_steps, ts_train_len - 1):
        X_train.append(ts_train_scaled[i - time_steps:i, :])
        y_train.append(ts_train_scaled[i:i + for_periods, :])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], feature_counts))

    inputs = all_data.values
    inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1, feature_counts)
    inputs = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i - time_steps:i, :])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], feature_counts))

    return X_train, y_train, X_test, sc

def LSTM_model(X_train, y_train, X_test, sc):
    # create a model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import SGD
    import os

    feature_counts = len(all_data.columns)

    # LSTM 아키텍쳐
    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(16,
                           return_sequences = True,
                           input_shape = (X_train.shape[1],feature_counts),
                           activation = 'tanh'))
    my_LSTM_model.add(LSTM(units = 16, activation = 'tanh'))
#     my_LSTM_model.add(Dropout(0.2))
    my_LSTM_model.add(Dense(feature_counts * 2, activation = 'relu'))
    my_LSTM_model.add(Dropout(0.2))
    my_LSTM_model.add(Dense(feature_counts))


    # Compiling
    #     sgd = optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)
    my_LSTM_model.compile(optimizer = SGD(learning_rate = 0.01, decay = 1e-7,
                                         momentum = 0.9, nesterov = False),
                         loss = 'mean_squared_error')
    # model 저장
    MODEL_DIR = './model/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    modelpath = MODEL_DIR + "{epoch:02d}-{val_loss:.4f}.hdf5"

    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                                   verbose=1, save_best_only=True)
    esc = EarlyStopping(monitor='val_loss', patience=10)

    # Fitting to the training set
    my_LSTM_model.fit(X_train, y_train, epochs = 500, batch_size = 32,validation_split = 0.2, callbacks = [esc], verbose = 1)
    # callbacks = [esc, checkpointer],
    LSTM_prediction = my_LSTM_model.predict(X_test)
    LSTM_prediction = sc.inverse_transform(LSTM_prediction)
#     print(LSTM_prediction)
    return my_LSTM_model, LSTM_prediction


def actual_pred_plot(preds):
    """
    Plot the actual vs prediction
    """
    actual_pred = pd.DataFrame(columns=['Cost', 'prediction'])
    actual_pred['Cost'] = all_data['2020':].iloc[:, -1][1:len(preds) + 1]
    actual_pred['prediction'] = preds[:, -1]

    from keras.metrics import MeanSquaredError
    m = MeanSquaredError()
    m.update_state(np.array(actual_pred['Cost']), np.array(actual_pred['prediction']))

    return m.result().numpy(), actual_pred.plot()

X_train, y_train, X_test, sc = ts_train_test_normalize(all_data, 6, 1)
my_lstm_model, lstm_predictions_2 = LSTM_model(X_train, y_train, X_test, sc)
actual_pred_plot(lstm_predictions_2)
plt.title('2nd_LSTM')
plt.xlabel('2021')

plt.ylabel('cost')
plt.show()
