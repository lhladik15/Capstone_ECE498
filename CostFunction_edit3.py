newPrices = pd.read_csv('APC_updates_new.csv', header=0, index_col='Date', parse_dates=True)
newPrices = newPrices['Close'].values

#the array we predicted
#array = [50, 20, 10, 60, 7
#our true values
true_array = [50, 20, 10, 2, 5]
#initial invest moneys
init_invest = 10000
invest = 10000
#initializing the counter i to keep track of the elements in our array for indexing
i = 1
#initializing the number of stocks we own
n = 0
#initializing value of cost of making a trade
m = 8.9

for x in range (0,len(array)-1):
    #to figure out if our algorithm thinks the stock is increasing/decreasing to buy or sell
    val = array[i]-array[(i-1)]
        #print(val)- used to see if our code was executing proper buys and sells
    if val>0:
        print('buy')
        n = int(invest/array[i])
        print('number stocks:', n)
        bought = invest*n

        if true_array[i]-true_array[i-1]>=0:
            profit_mult = 1
        else:
            profit_mult = -1
        bought = true_array[i] *n
        invest = (invest + bought) - m
        n = n
    elif val<0:
        print('sell')
        sold = n*true_array[i]
        invest = (invest + sold)-m
        n = 0
        print('number of stocks: ', n)
    else:
        print('hold')
    i = i +1
    invest = invest 

print('final value: ', invest)
percent_returns = ((invest-init_invest)/init_invest) * 100
print('percent returns: ', percent_returns)