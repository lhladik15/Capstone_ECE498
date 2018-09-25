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
import csv

#load data
def parser(x):
    return datetime.strptime(x, '%m/%d/%Y')
# load dataset
dataset = read_csv('APC_new1.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser= parser)
newPrices = read_csv('APC_updates_new.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser= parser)

a = newPrices.to_csv('newPrices')
b = read_csv('newPrices')
print('b: ', b)
