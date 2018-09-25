import pandas as pd
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error
from pandas import DataFrame

#df = pd.read_csv('EOG.csv',header=0,index_col='Date',parse_dates=True)
#newPrices = pd.read_csv('EOG_updates_new.csv',header=0,index_col='Date',parse_dates=True)

df = pd.read_csv('CHK.csv',header=0,index_col='Date',parse_dates=True)
newPrices = pd.read_csv('CHK_updates_new.csv',header=0,index_col='Date',parse_dates=True)

#newPrices.dropna()
#newPrices = newPrices[pd.notnull(newPrices['Close'])]
df = df[['Close']]
#print('df: ', df.tail())
newPrices = newPrices[['Close']]
#print('newPrices: ', newPrices)


forecast_out = int(10)
df['Label'] = df.shift(-forecast_out)
#print('df with label: ', df)

#Define features and labels
X = np.array(df.drop(['Label'],1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]
print('X_forecast: ', X_forecast)
print(X_forecast.shape)
X = X[:-forecast_out]
print('X: ', X)
print(X.shape)


y = np.array(df['Label'])
print('initial y: ', y)
print(y.shape)
y = y[:-forecast_out]
print('y new: ', y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size= 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print('forecast_prediction: ', forecast_prediction)
print(forecast_prediction.shape)
#prediction = pd.DataFrame(data=forecast_prediction)
#print(type(prediction))
#print(prediction)
#print(newPrices)

predicted_Vals = np.zeros((len(df)))
predicted_Vals.fill(np.nan)
predicted_Vals[-10:] = forecast_prediction
df['Predicted Values'] = predicted_Vals
true_future_Prices = np.zeros((len(df)))
true_future_Prices.fill(np.nan)
true_future_Prices[-10:] = newPrices['Close']
df['Real Prices'] =  true_future_Prices
format_col = df['Close']
format_col[-10:] = np.nan
df['Close'] = format_col
print(df)




#plotting
df['Close'].plot()
df['Predicted Values'].plot()
df['Real Prices'].plot()
plt.legend()
plt.title('CHK Stock Price History Versus Prediction')
plt.ylabel('Stock Price($)')
plt.xlabel('Date')
plt.show()







