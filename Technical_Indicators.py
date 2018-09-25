import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

df = pd.read_csv('APC.csv', header=0, index_col='Date', parse_dates=True)
df_close = df[['Close']]
df_open = df[['Open']]
df_volume = df[['Volume']]
df_high = df[['High']]
df_low = df[['Low']]

close_vals = df_close.values
close_vals = close_vals[:,0]
print(close_vals)
open_vals = df_open.values
open_vals = open_vals[:,0]
volume_vals = df_volume.values
volume_vals = volume_vals[:,0]
high_vals = df_high.values
high_vals = high_vals[:,0]
low_vals = df_low.values
low_vals = low_vals[:,0]

#EOG = EOG[['Close']]
#values = EOG.values
#values = values[:,0]

#calculate simple moving average
def SMA(values,window):
    weights = np.repeat(1.0,window)/window
    smas = np.convolve(values,weights,'valid')
    return smas

#calculate exponential moving average
def EMA(values, window):
    weights = np.exp(np.linspace(-1.,0.,window))
    weights /= weights.sum()

    a = np.convolve(values,weights)[:len(values)]
    a[:window] = a[window]
    return a

#Chande momentum oscillator
def cmo(prices, tf):
    CMO = []
    x = tf
    while x<len(prices):
        considerationPrices = prices[x-tf:x]
        upSum = 0
        downSum = 0
        y = 1
        while y < tf:
            currPrice = considerationPrices[y]
            prevPrice = considerationPrices[y-1]
            if currPrice >= prevPrice:
                upSum +=(currPrice-prevPrice)
            else:
                downSum += (prevPrice - currPrice)
            y += 1
        currCMO = ((upSum - downSum)/(upSum+float(downSum)))*100.00
        CMO.append(currCMO)
        x +=1

    return CMO

##Calculation of bollinger bands
#calculating our standard deviation for our bollinger bands calculation below
def standard_deviation(tf, prices):
    sd = []
    x = tf
    while x<=len(prices):
        array2consider = prices[x-tf:x]
        standdev = array2consider.std()
        sd.append(standdev)
        x += 1
    return  sd

def bollinger_bands(mult, tff):
     topBand = []
     botBand = []
     midband = []
     x = tff
     while x < len(close_vals):
         curSMA = SMA(close_vals[x-tff:x],tff)[-1]
         curSD = standard_deviation(tff,close_vals[0:tff])
         curSD = curSD[-1]
         TB = curSMA + (curSD*mult)
         BB = curSMA - (curSD*mult)
         topBand.append(TB)
         botBand.append(BB)
         midband.append(curSMA)
         x+= 1
     return topBand, botBand, midband

#Chaikin Money Flow
def CHMoF(c,h,l,o,v,tf):
    CHMF = []
    MFMs = []
    MFVs = []
    x = tf
    while x < len(c):
        PeriodVolume = 0
        volRange = v[x-tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol

        MFM = ((c[x]-l[x])-(h[x]-c[x]))/(h[x]-l[x])
        MFV = MFM * (PeriodVolume)

        MFMs.append(MFM)
        MFVs.append(MFV)
        x+=1

    y = tf
    while y < len(MFVs):
        PeriodVolume = 0
        volRange = v[x - tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol

        consider = MFVs[y-tf:y]
        tfsMFV = 0

        for eachMFV in consider:
            tfsMFV += eachMFV

        tfsCMF = tfsMFV/PeriodVolume
        CHMF.append(tfsCMF)

        y += 1

    return CHMF

#relative strength index
def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

####for calculating the Chaikin Volatility
def percentChange(startPoint, currentPoint):
    return((float(currentPoint)-startPoint)/abs(startPoint))*100.00
def chaikinVolCal(emaUsed, periodsAgo):
    chaikin_volatility = []
    highMlow = [] #for high minus low
    x = 0

    while x < len(close_vals):
        hml = high_vals[x] - low_vals[x]
        highMlow.append(hml)
        x +=1

    highMlowEMA = EMA(highMlow,emaUsed)

    y = emaUsed + periodsAgo

    while y < len(close_vals):
        cvc = percentChange(highMlowEMA[y-periodsAgo],highMlowEMA[y])

        chaikin_volatility.append(cvc)
        y +=1
    return chaikin_volatility

#Center of gravity
def cog(data,tf):
    COG = []
    x = tf

    while x<len(df_low):
        consider = data[x-tf:x]

        multipliers = range(1,tf+1)

        topFrac = 0
        botFrac = 0

        reversedOrder = reversed(consider)

        ordered = []
        for eachItem in reversedOrder:
            ordered.append(eachItem)
        for eachM in multipliers:
            addMe = eachM*ordered[eachM-1]
            addMe2 = ordered[eachM-1]

            topFrac += addMe
            botFrac += addMe2

        CeOfGr = -(topFrac/float(botFrac))

        COG.append(CeOfGr)

        x += 1
    return COG

#Rate of Change
def ROC(cp, tf):
    roc = []
    x = tf
    while x < len(close_vals):
        rocs = (cp[x]-cp[x-tf])/cp[x-tf]

        roc.append(rocs)
        x+=1

    return roc

#Average True Range
def TR(c,h,l,o,yc):
    x =  h - l
    y = abs(h-yc)
    z = abs(l-yc)

    if y<x>z:
        TR = x
    elif x <= y >= z:
        TR = y
    elif x <= z >=y:
        TR = z

    return TR

x = 1
TrueRanges = []
while x < len(close_vals):
    #TR(c,h,l,o,yc)
    TrueRange = TR(close_vals[x], high_vals[x], low_vals[x], open_vals[x], close_vals[x-1])
    TrueRanges.append(TrueRange)
    x += 1

ATR = EMA(TrueRanges, 14)
print(ATR)

#Average Directional Index
def DM(o,h,l,c,yo,yl,yc):
    moveUp = h - yh
    moveDown = yl - l

    if 0< moveUp > moveDown:
        PDM = moveUp
    else:
        PDM = 0

    if 0 < moveDown > moveUp:
        NDM = moveDown
    else:
        NDM = 0

    return PDM, NDM

def calcDIs():
    x = 1
    TrueRanges = []
    PosDMs = []
    NegDMs = []

    while x <len(close_vals):
        TrueRange =TR(close_vals[x], high_vals[x], low_vals[x], open_vals[x], close_vals[x-1])
        TrueRanges.append(TrueRange)

        PosDM, NegDM = DM(open_vals[x], high_vals[x], low_vals[x], close_vals[x], open_vals[x-1], high_vals[x-1], low_vals[x-1], close_vals[x-1])
        PosDMs.append(PosDM)
        NegDMs.append(NegDM)

        x += 1
    print (len(PosDMs))

    expPosDM = EMA(PosDMs,14)
    expNegDM = EMA(NegDMs, 14)
    ATR = EMA(TrueRanges, 14)

    xx = 0
    PDIs = []
    NDIs = []

    while xx< len(ATR):
        PDI = 100 *(expPosDM[xx]/ATR[xx])
        PDIs.append(PDI)

        NDI = 100*(expNegDM[xx]/ATR[xx])
        NDIs.append(NDI)

        xx += 1

    return PDIs, NDIs

def ADX():
    PositiveDI, NegativeDI = calcDIs()

    print(len(PositiveDI))
    print(len(NegativeDI))

    xxx = 0
    DXs = []

    while xxx < len(close_vals):
        DX = 100*((abs(PositiveDI[xxx]-NegativeDI[xxx])/(PositiveDI[xxx]+NegativeDI[xxx])))

        DXs.append(DX)
        xxx += 1

    print(len(DXs))

    ADX = EMA(DXs, 14)

    print(len(ADX))
    print (ADX)

ADX()













a = ROC(close_vals,30)
print(a)










sma = SMA(close_vals,44)
ema = EMA(close_vals,3)
CMO = cmo(close_vals, 10)
tb, bb, md = bollinger_bands(2,20)
CHMOF = CHMoF(close_vals, high_vals,low_vals,open_vals,volume_vals,20)
RSI = rsiFunc(close_vals)

print(len(sma))
print(len(ema))
print(len(CMO))
print(len(RSI))
print(len(CHMOF))

#plt.plot(sma)
#plt.plot(ema)
#plt.plot(CMO)
plt.plot(RSI)


#plotting bollinger bands
#plt.plot(bb)
#plt.plot(md)
#plt.plot(tb)

plt.show()

