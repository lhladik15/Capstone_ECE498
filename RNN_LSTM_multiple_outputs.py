from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from math import sqrt
import numpy as np
import pandas as pd
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


#load data
def parser(x):
    return datetime.strptime(x, '%m/%d/%Y')
# load dataset
dataset = read_csv('APC_new1.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser= parser)
#print(type(dataset))
print(dataset.head())
values = dataset.values

#print(dataset.head())

######PLOTTING OUR INDIVIDUAL FEATURES
#specify columns to plot
#groups = [0, 1, 2, 3, 4, 5,6]
#i = 1
#plot each column
#pyplot.figure()
#for group in groups:
    #pyplot.subplot(len(groups), 1, i)
    #pyplot.plot(values[:,group])
    #pyplot.title(dataset.columns[group], y=0.5, loc='right')
    #i += 1
#pyplot.show()

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('APC_new1.csv', header=0, index_col=0)
print('Dataset:', dataset.head())
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(reframed.head())
#print(len(reframed['var1(t)']))

# split into train and test sets
values = reframed.values
n_train_days = int(0.99*len(reframed['var1(t)']))
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, :-1]
test_X, test_y = test[:, :-1], test[:, :-1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print('Atention: ', train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(7))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.xlabel('Epoch')
#pyplot.ylabel('Loss')
#pyplot.title('Loss Throughout Training')
#pyplot.show()

# make a prediction
yhat = model.predict(test_X)
#print('yhat shape: ',yhat.shape)
#print('yhat ', yhat)
test_X = test_X.reshape((8,7))
#print('X_test', test_X.shape)
# invert scaling for forecast

#inv_yhat = concatenate([yhat, test_X])
inv_yhat = scaler.inverse_transform(yhat)
#print('inv_yhat: ', inv_yhat.shape)
#print(inv_yhat)

#a = inv_yhat[:,0] #prediction for one intput variable
#print(a.shape)
#print(a)

inv_yhat = inv_yhat.reshape((inv_yhat.shape[0], 1, inv_yhat.shape[1]))
newout = model.predict(inv_yhat)
print('LOOK: ', newout)

# invert scaling for actual
#test_y = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
#inv_y = inv_y[:,0]
# calculate RMSE
#rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#print('Test RMSE: %.3f' % rmse)
#print(len(inv_y))
#print(len(inv_yhat))




#predicted_Vals = np.zeros((len(dataset['Views'])))
#predicted_Vals.fill(np.nan)
#predicted_Vals[-n_train_days:] = inv_yhat
#print(predicted_Vals)
#print(len(predicted_Vals))
#print(len(dataset['Views']))
#print(type(predicted_Vals)
#predicted_Vals = pd.DataFrame(predicted_Vals)
#result = pd.concat([dataset,predicted_Vals])
#result = dataset
#result['Prediction'] = predicted_Vals
#print('Result:', result)




