import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error

newPrices = pd.read_csv('EOG_updates.csv',header=0,index_col='Date',parse_dates=True)
newPrices = newPrices[['Close']]

APA = pd.read_csv('EOG.csv',header=0,index_col='Date',parse_dates=True)
APC = pd.read_csv('APC.csv',header=0,index_col='Date',parse_dates=True)
#CHK = pd.read_csv('CHK.csv',header=0,index_col='Date',parse_dates=True)
#COG = pd.read_csv('COG.csv',header=0,index_col='Date',parse_dates=True)
#COP = pd.read_csv('COP1.csv',header=0,index_col='Date',parse_dates=True)
#CVX = pd.read_csv('CVX.csv',header=0,index_col='Date',parse_dates=True)
#DVN = pd.read_csv('DVN.csv',header=0,index_col='Date',parse_dates=True)
EOG = pd.read_csv('EOG.csv',header=0,index_col='Date',parse_dates=True)
#EQT = pd.read_csv('EQT1.csv',header=0,index_col='Date',parse_dates=True)
#HAL = pd.read_csv('HAL1.csv',header=0,index_col='Date',parse_dates=True)
#HES = pd.read_csv('HES1.csv',header=0,index_col='Date',parse_dates=True)
#HP = pd.read_csv('HP1.csv',header=0,index_col='Date',parse_dates=True)
#KMI = pd.read_csv('KMI1.csv',header=0,index_col='Date',parse_dates=True)
#MPC = pd.read_csv('MPC1.csv',header=0,index_col='Date',parse_dates=True)
#MRO = pd.read_csv('MRO1.csv',header=0,index_col='Date',parse_dates=True)
#MUR = pd.read_csv('MUR1.csv',header=0,index_col='Date',parse_dates=True)
#NBL = pd.read_csv('NBL1.csv',header=0,index_col='Date',parse_dates=True)
#NFX = pd.read_csv('NFX1.csv',header=0,index_col='Date',parse_dates=True)
#NOV = pd.read_csv('NOV1.csv',header=0,index_col='Date',parse_dates=True)
#OKE = pd.read_csv('OKE1.csv',header=0,index_col='Date',parse_dates=True)
#OXY = pd.read_csv('OXY1.csv',header=0,index_col='Date',parse_dates=True)
##PSX = pd.read_csv('PSX2',header=0,index_col='Date',parse_dates=True)
#PXD = pd.read_csv('PXD1.csv',header=0,index_col='Date',parse_dates=True)
#RRC = pd.read_csv('RRC1.csv',header=0,index_col='Date',parse_dates=True)
#SLB = pd.read_csv('SLB1.csv',header=0,index_col='Date',parse_dates=True)
#VLO = pd.read_csv('VLO1.csv',header=0,index_col='Date',parse_dates=True)
#WMB = pd.read_csv('WMB1.csv',header=0,index_col='Date',parse_dates=True)
#XEC = pd.read_csv('XEC1.csv',header=0,index_col='Date',parse_dates=True)
#XOM = pd.read_csv('XOM1.csv',header=0,index_col='Date',parse_dates=True)

APA = APA[['Close']]
APC = APC[['Close']]
#CHK = CHK[['Close']]
#COG = COG[['Close']]
#COP = COP[['Close']]
#CVX = CVX[['Close']]
#DVN = DVN[['Close']]
#EOG = EOG[['Close']]
EOG = EOG[['Close']]
#EQT = EQT[['Close']]
#HAL = HAL[['Close']]
#HES = HES[['Close']]
#HP = HP[['Close']]
#KMI = KMI[['Close']]
#MPC = MPC[['Close']]
#MRO = MRO[['Close']]
#MUR = MUR[['Close']]
#NBL = NBL[['Close']]
#NFX = NFX[['Close']]
#NOV = NOV[['Close']]
#OKE = OKE[['Close']]
#OXY = OXY[['Close']]
##PSX = PSX[['Close']]
#PXD = PXD[['Close']]
#RRC = RRC[['Close']]
#SLB = SLB[['Close']]
#VLO = VLO[['Close']]
#WMB = WMB[['Close']]
#XEC = XEC[['Close']]
#XOM = XOM[['Close']]

#plot closing prices of all energy sector stocks
#APA['Close'].plot(grid=True)
#APC['Close'].plot(grid=True)
#CHK['Close'].plot(grid=True)
#COG['Close'].plot(grid=True)
#COP['Close'].plot(grid=True)
#CVX['Close'].plot(grid=True)
#DVN['Close'].plot(grid=True)
#EOG['Close'].plot(grid=True)
#EQT['Close'].plot(grid=True)
#EOG['Close'].plot(grid=True)
#APA ['Close'].plot(grid=True)
#EQT['Close'].plot(grid=True)
#HAL['Close'].plot(grid=True)
#HES['Close'].plot(grid=True)
#HP['Close'].plot(grid=True)
#KMI['Close'].plot(grid=True)
#MPC['Close'].plot(grid=True)
#MRO['Close'].plot(grid=True)
#MUR['Close'].plot(grid=True)
#NBL['Close'].plot(grid=True)
#NFX['Close'].plot(grid=True)
#NOV['Close'].plot(grid=True)
#OKE['Close'].plot(grid=True)
#OXY['Close'].plot(grid=True)
##PSX['Close'].plot(grid=True)
#PXD['Close'].plot(grid=True)
#RRC['Close'].plot(grid=True)
#SLB['Close'].plot(grid=True)
#VLO['Close'].plot(grid=True)
#WMB['Close'].plot(grid=True)
#XEC['Close'].plot(grid=True)
#XOM['Close'].plot(grid=True)

#plt.title('Historical Stock Data EOG and APC')
#plt.title('Energy Sector Historical Stock Prices')
#plt.ylabel('Stock Price')
#plt.show()


#EOG = EOG[['Close']]
forecast_out = int(10)
EOG['Prediction'] = EOG[['Close']].shift(-forecast_out)
#print(EOG['Prediction'])


#Defining features and labels
X = np.array(EOG.drop(['Prediction'],1))
X = preprocessing.scale(X)
#X_forecast = X[-forecast_out:] #set X_forecast to last 30
X_forecast = X[-forecast_out:]
X = X[:-forecast_out] #remove last 30 from X

###FOR PREDICTING APC USING EOG STOCK
X_APC = np.array(APC)
X_predict_APC = X_APC[-forecast_out:]

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
forecast_prediction = clf.predict(X_forecast) ##preding EOG
#forecast_prediction_APC = clf.predict(X_predict_APC) ######APC!!!!!!!!


#print(forecast_prediction)

#predicted_Vals = np.zeros((len(EOG)))
predicted_Vals = np.zeros((len(EOG)))
predicted_Vals.fill(np.nan)
#predicted_Vals[-10:] = forecast_prediction
#predicted_Vals[-10:] = forecast_prediction_APC ######APC!!!!!!!!!!!!!!!!!!
#predicted_Vals[-(len(newPrices):] = forecast_prediction
#EOG['Predicted Values'] = predicted_Vals


##plot true data and prediction
#EOG['Close'].plot(grid=True)
#EOG['Predicted Values'].plot(grid = True)
#EOG_close_true = EOG['Close']
#EOG_Predicted = EOG['Predicted Values']
#EOG_close_true[-40:].plot(grid = True)
#EOG_Predicted[-40:].plot(grid=True)

APC_close_true = APC['Close']
APC_Predicted = EOG['Prediction']
APC_close_true[-40:].plot(grid = True)
APC_Predicted[-40:].plot(grid=True)

plt.ylabel('Stock Price')
plt.title('True Stock Data and 30 Day Forecast')
plt.show()

#y = EOG[-10:]
#y_true = y['Close']
#y_pred = y['Predicted Values']
#mse = mean_squared_error(y_true,y_pred)

print("mean squared error:", mse)

#------------------------------------------------
#TECHNICAL INDICATOR

#EOG['Moving Average'] = EOG['Close'].rolling(window=44).mean()

#ax1 = plt.subplot2grid((6,1),(0,0), rowspan = 5, colspan= 1)
#ax2 = plt.subplot2grid((6,1),(5,0), rowspan = 1, colspan= 1)

#ax1.plot(EOG.index, EOG['Close'])
#ax1.plot(EOG.index, EOG['Moving Average'])
#ax2.bar(EOG.index, EOG['Volume'])

#EOG['Moving Average'].plot(grid=True)
#EOG['Close'].plot(grid=True)

#plt.show()
