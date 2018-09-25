import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
import math



# df = pd.read_csv('EOG.csv',header=0,index_col='Date',parse_dates=True)
# newPrices = pd.read_csv('EOG_updates_new.csv',header=0,index_col='Date',parse_dates=True)

df = pd.read_csv('APC.csv', header=0, index_col='Date', parse_dates=True)
newPrices = pd.read_csv('APC_updates_new.csv', header=0, index_col='Date', parse_dates=True)

# newPrices.dropna()
# newPrices = newPrices[pd.notnull(newPrices['Close'])]
df = df[['Close']]
# print('df: ', df.tail())
newPrices = newPrices[['Close']]
# print('newPrices: ', newPrices)


forecast_out = int(10)
# make sure our new Prices array fits with our prediction for plotting (for values other than 10)
newPrices = newPrices.iloc[0:forecast_out]
print('newPrices: ', newPrices)
df['Label'] = df.shift(-forecast_out)
# df = df.iloc[:-3] #needed for APA updates because we deleted the last three rows-->need to eliminate NaN
print('df with label: ', df)

# Define features and labels
X = np.array(df.drop(['Label'], 1))
print('X: ', X)
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

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print('forecast_prediction: ', forecast_prediction)
print('type: ', type(forecast_prediction))
print(forecast_prediction.shape)
# prediction = pd.DataFrame(data=forecast_prediction)
# print(type(prediction))
# print(prediction)
# print(newPrices)

predicted_Vals = np.zeros((len(df)))
predicted_Vals.fill(np.nan)
predicted_Vals[-forecast_out:] = forecast_prediction
df['Predicted Values'] = predicted_Vals
true_future_Prices = np.zeros((len(df)))
true_future_Prices.fill(np.nan)
true_future_Prices[-forecast_out:] = newPrices['Close']
df['Real Prices'] = true_future_Prices
# print('true future prices: ', true_future_Prices)
true_Prices = true_future_Prices[-forecast_out:]
# print('True Prices: ', true_Prices)
print('Forecast pred: ', forecast_prediction)
format_col = df['Close']
format_col[-forecast_out:] = np.nan
df['Close'] = format_col
print('Output dataframe: ', df)
# calculating error
rmse = sqrt(mean_squared_error(forecast_prediction, true_Prices))
print('Test RMSE: %.4f' % rmse)

# plotting
# df['Close'].plot()
# df['Predicted Values'].plot()
# df['Real Prices'].plot()
# plt.legend()
# plt.title('Stock Price History Versus Prediction')
# plt.ylabel('Stock Price($)')
# plt.xlabel('Date')
# plt.show()

# the array we predicted
array = forecast_prediction
print('array: ', array)
print(array.shape)
# our true values
true_array = df['Real Prices'].iloc[-(forecast_out+1):]
true_array = true_array.values
print('true array: ', true_array)
print(true_array.shape)
# initial invest moneys
init_invest = 10000
invest = 10000
# initializing the counter i to keep track of the elements in our array for indexing
i = 1

# initializing the number of stocks we own in array format to keep track of our stocks
n = np.zeros(len(array)-1)

# initializing array of zeros to hold value of our stock holdings
holdings = np.zeros(len(array)-1)  # the monetary stock we bought based on our prediction (for n value)
# initialized as an empty array

# initializing value of cost of making a trade
m = 8.9

# initializing array with money value
money = np.zeros(len(array)-1)
money[0] = init_invest
print('money: ', money)



for x in range(1, len(array)-1):

    order = array[x+1] - array[x]
    print('#########################')
    print('order: ', order)
    print('money: ', money[x-1])


    # buy order condition (need at least 500 to invest and must predict that stock will increase)
    if ((money[x-1]) >= 500 and order > 0):
        print('buy')
        if x ==0:
            x = 1
        # creating value for number of stocks to buy
        n[x] = math.floor((money[x-1]) / true_array[x])  # buy number of stocks based on our predicted value
        # calculate the actual value of stocks being bought (must use real price of stock)
        print('n: ', n[x])
        buy = float(n[x] * true_array[x])
        print('true array: ', true_array)
        print('bought: ', buy)
        # subtract amount paid from our money funds (subtract amount of stock bought and amount paid to execute transaction)
        money[x] = money[x-1] - buy
        # stock holdings value (add new holds to previous holdings)
        holdings[x] = holdings[x-1] + buy
    elif (money[x]<500 and order >= 0):
        print('hold')
        # record the value of our holdings
        if x ==0:
            x = 1
        holdings[x] = n[x-1] * true_array[x]
        # record the amount of stocks we are currently holding (this number has not changed from previous day)
        n[x] = n[x - 1]
        # record the amount of leftover money
        money[x] = money[x - 1]
    elif (order < 0):
        print('sell')
        if x ==0:
            x = 1
        # value of stock holdings goes to zero since we are selling all
        holdings[x] = 0
        # value of stocks we are currently holding also goes to 0
        n[x] = 0
        # record the amount of money we have "leftover"
        money[x] = (holdings[x-1]) + money[x-1]
    elif (array[i] == array[i-1]):
        print('hold')
        if x ==0:
            x = 1
        money[x] = money[x-1]
        n[x] = n[x-1]
        holdings = holdings[x-1]

    # use i as a counter
    i += 1

    print('stock holdings: ', holdings[x])
    print('stocks held: ', n[x])
    print('money left: ', money[x])

result = holdings[-1] + money[-1]
percent_return = ((result - init_invest)/init_invest)*100
print(result)
print(percent_return)
portfolio_value = holdings + money
print('portfolio: ', portfolio_value)