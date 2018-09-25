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
print('Dataset:', dataset), dataset.shape
values = dataset.values
print('values: ', values)
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 365, 10)
print('Reframed:', reframed)
print('Reframed shape: ', reframed.shape)
print(len(reframed['var1(t)']))
print('Type: ', type(reframed))

#slicing our dataframe to get train and test sets:
#previous time steps (our raw data)
X = reframed.iloc[:, 0:2555]
#print('AAAAAA: ', train)
#print('shape train: ', train.shape)
#future time steps (for future data)
y  = reframed.iloc[:, 2555:2625]
#print('BBBBB: ', test)
#print('shape test: ', test.shape)
#convert our values into numpy array:
X = X.values
#print('Values train: ', train.shape)
y = y.values
#print('Values test: ', test.shape)

#split into inputs and outputs
train_X = X[0:395,:]
train_y = y[0:395,:]
test_X = X[396:405,:]
test_y = y[396:405,:]
n_days = 10
n_features = 7

print('shape train_X: ', train_X.shape, train_X)
print('shape train_y: ', train_y.shape, train_y)
print('shape test_X: ', test_X.shape, test_X)
print('shape test_y: ', test_y.shape, test_y)
#reshape to be 3D [samples, timesteps, features
train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days,n_features))

print('reshaped stuff', train_X.shape)
print('reshaped stuff', test_X.shape)


#design network:
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss = 'mae', optimizer= 'adam')
#fit network
history = model.fit(train_X, train_y, epochs= 1, batch_size= 72, validation_data=(test_X,test_y), verbose=2, shuffle = False)

yhat = model.predict(test_X)
#print('prediction: ', yhat)
print('prediction shape: ', yhat.shape)




