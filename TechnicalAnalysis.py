#This module defines function for calculating technical indicators from the stock data
#Author: Luckyson Khaidem
#Date: 21/1/16

import numpy as np

#exponential weighted moving average
def ema(x,p):

	x = np.squeeze(x)
	prev_ema = np.mean(x[:p])
	ema = [prev_ema]
	m = len(x)
	multiplier = 2/float(p+1)
	for i in range(p,m):
		cur_ema = (x[i] - prev_ema)*multiplier + prev_ema
		prev_ema = cur_ema
		ema.append(cur_ema)
	return np.array(ema)

#Relative Strength Index
def getRSI(x):
	x = np.squeeze(x)
	n = len(x)
	x0 = x[:n-1]
	x1 = x[1:]
	change = x1 - x0
	avgGain = []
	avgLoss = []
	loss = 0
	gain = 0
	for i in range(14):
		if change[i] > 0 :
			gain += change[i]
		elif change[i] < 0:
			loss += abs(change[i])
	averageGain = gain/14.0
	averageLoss = loss/14.0
	avgGain.append(averageGain)
	avgLoss.append(averageLoss)
	for i in range(14,n-1):
		if change[i] >= 0:
			avgGain.append((avgGain[-1]*13+change[i])/14.0)
			avgLoss.append((avgLoss[-1]*13)/14.0)
		else:
			avgGain.append((avgGain[-1]*13)/14.0)
			avgLoss.append((avgLoss[-1]*13+abs(change[i]))/14.0)
	avgGain = np.array(avgGain)
	avgLoss = np.array(avgLoss)
	RS = avgGain/avgLoss
	RSI = 100 -(100/(1+RS))

	return np.c_[RSI,x1[13:]]

def getStochasticOscillator(x):

	high = np.squeeze(x[:,1])
	low = np.squeeze(x[:,2])
	close = np.squeeze(x[:,3])
	n = len(high)
	highestHigh = []
	lowestLow = []
	for i in range(n-13):
		highestHigh.append(high[i:i+14].max())
		lowestLow.append(low[i:i+14].min())
	highestHigh = np.array(highestHigh)
	lowestLow = np.array(lowestLow)
	k = 100*((close[13:]-lowestLow)/(highestHigh-lowestLow))

	return np.c_[k,close[13:]]

def getWilliams(x):

	high = np.squeeze(x[:,1])
	low = np.squeeze(x[:,2])
	close = np.squeeze(x[:,3])
	n = len(high)
	highestHigh = []
	lowestLow = []
	for i in range(n-13):
		highestHigh.append(high[i:i+14].max())
		lowestLow.append(low[i:i+14].min())
	highestHigh = np.array(highestHigh)
	lowestLow = np.array(lowestLow)
	w = -100*((highestHigh-close[13:])/(highestHigh-lowestLow))

	return np.c_[w,close[13:]]

def getMACD(close):

	ma1 = ema(np.squeeze(close),12)
	ma2 = ema(np.squeeze(close),26)
	macd =  ma1[14:] - ma2
	
	return np.c_[macd,close[len(close) - len(macd):]]

def getPriceRateOfChange(close,n_days):

	n_days = int(n_days)
	close = np.squeeze(close)
	n = len(close)
	x0 = close[:n-n_days]
	x1 = close[n_days:]
	PriceRateOfChange = (x1 - x0)/x0
	
	return np.c_[PriceRateOfChange,x1]

def getOnBalanceVolume(X):
	
	close = np.squeeze(X[:,3])
	volume = np.squeeze(X[:,4])[1:]
	n = len(close)
	x0 = close[:n-1]
	x1 = close[1:]
	change = x1 - x0
	OBV = []
	prev_OBV = 0

	for i in range(n-1):
		if change[i] > 0:
			current_OBV = prev_OBV + volume[i]
		elif change[i] < 0:
			current_OBV = prev_OBV - volume[i]
		else:
			current_OBV = prev_OBV
		OBV.append(current_OBV)
		prev_OBV = current_OBV
	OBV = np.array(OBV)        
	
	return np.c_[OBV,x1]







