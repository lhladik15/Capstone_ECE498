import alpha_vantage
import pandas
from alpha_vantage.timeseries import TimeSeries

#ts = TimeSeries(key='Q95GS8TO0UFFXD9FR', output_format= 'pandas', indexing_type='date')
ts = TimeSeries(key='Q95GS8TO0UFFXD9FR', output_format= 'pandas', indexing_type='date')

data, meta_data = ts.get_daily('APC')

print(data)