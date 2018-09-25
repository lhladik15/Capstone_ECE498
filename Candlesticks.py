
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import pandas as pd
from mpl_finance import candlestick_ochl

#style.use('ggplot')

df = pd.read_csv('CHK.csv', parse_dates=True, index_col=0)

df_ohlc = df['Adj Close'].resample('5D').ohlc()
df_volume = df['Volume'].resample('5D').sum()
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ochl(ax1, df_ohlc.values, width=5,colordown='red')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()