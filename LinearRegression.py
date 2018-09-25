import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error
from mpl_finance import candlestick_ochl

EOG = pd.read_csv('EOG.csv',header=0,index_col='Date',parse_dates=True)
EOG = EOG[['Close']]
forecast_out = int(30)
EOG['Prediction'] = EOG[['Close']].shift(-forecast_out)

#Defining features and labels
X = np.array(EOG.drop(['Prediction'],1))
X = preprocessing.scale(X)
X_forecast = X[-forecast_out:] #set X_forecast to last 30
X = X[:-forecast_out] #remove last 30 from X

y = np.array(EOG['Prediction']) #defining y output as equal to array of prediction values
y = y[:-forecast_out] #remove last 30 days

#Linear Regression
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#training
clf = LinearRegression()
clf.fit(X_train,y_train)
#testing
confidence = clf.score(X_test, y_test)
print("confidence:", confidence)

#Predict our X_forecast values:
forecast_prediction = clf.predict(X_forecast)
#print(forecast_prediction)

predicted_Vals = np.zeros((len(EOG)))
predicted_Vals.fill(np.nan)
predicted_Vals[-30:] = forecast_prediction
EOG['Predicted Values'] = predicted_Vals

EOG['Close'].plot(grid=True)
EOG['Predicted Values'].plot(grid = True)

y = EOG[-30:]
y_true = y['Close']
y_pred = y['Predicted Values']
mse = mean_squared_error(y_true,y_pred)

print("meand squared error:", mse)

####technical indicators:
#moving average
EOG['Moving Average'] = EOG['Close'].rolling(window=44).mean()



ax1 = plt.subplot2grid((6,1),(0,0), rowspan = 5, colspan= 1)
ax2 = plt.subplot2grid((6,1),(5,0), rowspan = 1, colspan= 1)

ax1.plot(EOG.index, EOG['Close'])
ax1.plot(EOG.index, EOG['Moving Average'])
#ax2.bar(EOG.index, EOG['Volume'])

EOG['Moving Average'].plot(grid=True)
EOG['Close'].plot(grid=True)

plt.show()




