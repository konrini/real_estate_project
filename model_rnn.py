import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
name = '모델링 마트 테이블1.csv'
table = pd.read_csv(name, encoding = 'utf-8')


for i in range(len(table['date'])):
    table['date'][i] = ''.join(table['date'][i].split('-'))
    table['date'][i] = datetime.strptime(str(table['date'][i]), "%Y%m").date()

test1 = table.interpolate(method = 'polynomial', order = 3)
test1.set_index('date', inplace = True)
test1 = test1.dropna()


def ts_train_test_normalize(all_data, time_steps = 7, for_periods = 1):
    """
    input:
        data: dataframe with dates and price data
    output:
        X_train, y_train: data from 2013/1/1-2018/12/31
        X_test : data from 2019-
        sc :     insantiated MinMaxScaler object fit to the training data
    """
    # create training and test set
    ts_train = all_data.iloc[:-12, :].values
    ts_test = all_data.iloc[-12:, :].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = sc.fit_transform(ts_train)

    # create training data of s samples and t time steps
    X_train = []
    y_train = []
    for i in range(time_steps, ts_train_len-1):
        X_train.append(ts_train_scaled[i-time_steps:i, :])
        y_train.append(ts_train_scaled[i:i+for_periods, :])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # # Reshaping X_train for efficient modelling
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 6))

    inputs = all_data.values
    inputs = inputs[len(inputs)-len(ts_test)-time_steps:]
    inputs = inputs.reshape(-1, 6)
    inputs = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i-time_steps:i, :])

    X_test = np.array(X_test)
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 6))

    return X_train, y_train, X_test, sc


def simple_rnn_model(X_train, y_train, X_test, sc):
    """
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    """
    # create a model
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    import os

    my_rnn_model = Sequential()
    my_rnn_model.add(SimpleRNN(32, return_sequences = True))
    my_rnn_model.add(SimpleRNN(32))
    my_rnn_model.add(Dense(6))  # The time step of the output

    my_rnn_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    MODEL_DIR = './model/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    modelpath = MODEL_DIR + "{epoch:02d}-{val_loss:.4f}.hdf5"

    checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                                   verbose=1, save_best_only=True)
    esc = EarlyStopping(monitor='val_loss', patience=5)

    #     # fit the RNN model
    my_rnn_model.fit(X_train, y_train, epochs = 1000, batch_size = 10, validation_split = 0.2, callbacks = [esc, checkpointer], verbose = 0)

    # Finalizing predictions
    rnn_predictions = my_rnn_model.predict(X_test)
    from sklearn.preprocessing import MinMaxScaler
    rnn_predictions = sc.inverse_transform(rnn_predictions)

    return my_rnn_model, rnn_predictions


def actual_pred_plot(preds):
    """
    Plot the actual vs prediction
    """
    actual_pred = pd.DataFrame(columns=['Cost', 'prediction'])
    actual_pred['Cost'] = test1.iloc[-12:, -1][0:len(preds)]
    actual_pred['prediction'] = preds[:, -1]

    from keras.metrics import MeanSquaredError
    m = MeanSquaredError()
    m.update_state(np.array(actual_pred['Cost']), np.array(actual_pred['prediction']))

    return m.result().numpy(), actual_pred.plot()


X_train, y_train, X_test, sc = ts_train_test_normalize(test1)
my_rnn_model, rnn_predictions_2 = simple_rnn_model(X_train, y_train, X_test, sc)
rnn_predictions_2 = rnn_predictions_2[1:]
actual_pred_plot(rnn_predictions_2)
plt.show()

