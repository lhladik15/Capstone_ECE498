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


newPrices = pd.read_csv('APC_updates_new.csv', header=0, index_col='Date', parse_dates=True)

#array = [40, 10, 50, 60, 70, 30, 25, 40]
array = newPrices['Close'].values
init_invest = 10000
invest = 10000
#initializing cost for executing a trade:
m = 8.90
#initializing the counter i to keep track of the elements in our array for indexing
i = 1
#initialize number of stocks we are going to buy
n = math.floor((init_invest)/array[0]) #only buy as much as we can the first day of investing
print('n: ', n)
#leftover funds
leftover = init_invest - (n*array[0])
#initialize an array to put our stock values in
stock_tracker = np.zeros([len(array)-1,1])
#initialzie counter for our stock_tracker numpy array
y = 0

for x in range (0,len(array)-1):

    if (array[i]-array[(i-1)])<0:
        print('loss')

    else:
        print('profit')


    stock_tracker[y] = n*array[i]
    i = i+1
    y = y+1

print('stock_tracker: ', stock_tracker)


final_result = stock_tracker[-1]+leftover
print('final result: ', final_result)

percent_change = ((final_result-init_invest)/init_invest)*100
print('percent_change: ', percent_change)

portfolio_value = stock_tracker + leftover
print('portfolio: ', portfolio_value)



