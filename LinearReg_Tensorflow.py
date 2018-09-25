import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#SPY = pd.read_csv('SPY.csv')
#SPY.index = pd.to_datetime(SPY.index)
APA = pd.read_csv('APA1.csv')
APA.index = pd.to_datetime(APA.index)
APC = pd.read_csv('APC2.csv')
APC.index = pd.to_datetime(APC.index)
CHK = pd.read_csv('CHK1.csv')
CHK.index = pd.to_datetime(CHK.index)
COG = pd.read_csv('COG1.csv')
COG.index = pd.to_datetime(COG.index)
CVX = pd.read_csv('CVX1.csv')
CVX.index = pd.to_datetime(CVX.index)
DVN = pd.read_csv('DVN1.csv')
DVN.index = pd.to_datetime(DVN.index)
EOG = pd.read_csv('EOG1.csv')
EOG.index = pd.to_datetime(DVN.index)



#SPY['Close'].plot()
#APA['Close'].plot()
#APC['Close'].plot()
#CHK['Close'].plot()
#COG['Close'].plot()
#CVX['Close'].plot()
#DVN['Close'].plot()
#EOG['Close'].plot()
#plt.show()



forecast_col = 'Close'

#do this because machine learning cannot work with nan data, so want algorithm to treat missing data
#data as outlier
APA.fillna(-9999, inplace=True)
#using ceiling to get an integer value
#this is the number of days out we are forecasting (10% of dataframe)
forecast_out = int(math.ceil(0.1*len(APA)))


#now we need labels:
APA['label']=APA[forecast_col].shift(-forecast_out)
#above line shifts our column up, so each row for adjusted column will be adjusted price
#for 10 days into the future (features are attributes of what we think will cause adjusted
#close price to change

X = np.array(APA.drop(['label'],1)) #features
X=X[:-forecast_out]
X_lately = X[-forecast_out:]
#X = preprocessing.scale(X) #this step slows down your runtime, so likely skip with high frequency stock trading

APA.dropna(inplace=True)
y = np.array(APA['label'])

#print(len(X),len(y))

#create our training and testing set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train) #for training
accuracy = clf.score(X_test, y_test) #for testing
#note: have to train and test with separate data!

print(accuracy)

#note to do SVR: clf.SVR()

#now let's do prediction:

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
# we did those to get our date values (which were not in features or labels)

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix +=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]

APA['Close'].plot()
plt.show()
APA['Forecast'].plot()

plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
print('end')





















#ql.ApiConfig.api_key = "unfauMGCTgKycpYeB-Mf"
#data = ql.get_table("AS/UTIL", paginate=True)
#data = ql.get_table("AS/SP500", paginate=True)


#APC = data.loc[data['ticker']=='APC']
#CHK = data.loc[data['ticker']=='CHK']
#COG = data.loc[data['ticker']=='COG']
#COP = data.loc[data['ticker']=='COP']
#CVX = data.loc[data['ticker']=='CVX']
#DVN = data.loc[data['ticker']=='DVN']
#EOG = data.loc[data['ticker']=='EOG']
#EQT = data.loc[data['ticker']=='EQT']
#FTI = data.loc[data['ticker']=='FTI']
#HAL = data.loc[data['ticker']=='HAL']
#HES = data.loc[data['ticker']=='HES']
#HAL = data.loc[data['ticker']=='HAL']
#HP = data.loc[data['ticker']=='HP']
#KMI = data.loc[data['ticker']=='KMI']
#MPC = data.loc[data['ticker']=='MPC']
#MRO = data.loc[data['ticker']=='MRO']
#MUR = data.loc[data['ticker']=='MUR']
#NBL = data.loc[data['ticker']=='NBL']
#NFX = data.loc[data['ticker']=='NFX']
#NOV = data.loc[data['ticker']=='NOV']
#OKE = data.loc[data['ticker']=='OKE']
#OXY = data.loc[data['ticker']=='OXY']
#PSX = data.loc[data['ticker']=='PSX']
#PXD = data.loc[data['ticker']=='PXD']
#RRC = data.loc[data['ticker']=='RRC']
#SLB = data.loc[data['ticker']=='SLB']
#VLO = data.loc[data['ticker']=='VLO']
#WMB = data.loc[data['ticker']=='WMB']
#XEC = data.loc[data['ticker']=='XEC']
#XEC = data.loc[data['ticker']=='XOM']

#import matplotlib.pyplot as plt
#APA['close'].plot()
#plt.show()

#print(len(APA))
#axis = np.linspace(0,390,391)
#plt.plot(axis,APA['low'])

#EXC = data.loc[data['ticker']=='FE']
#print(EXC)

#print(len(data))







