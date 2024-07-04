from pandas import read_csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from TechnicalAnalysis import *
from DataFetcher import DataFetcher
import os
from datetime import datetime
from matplotlib import pyplot as plt
from DataPreprocessor import DataPreprocessor	
from ModelEvaluation import Evaluator
from multiprocessing import Process 
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.model_selection import cross_val_score

def getData(data):
	
	smoother = DataPreprocessor()
	#data = data[::-1] # reverse
	ohclv_data = np.c_[data['Open'],
					   data['High'],
					   data['Low'],
					   data['Close'],
					   data['Volume']]
	ohclv_df = pd.DataFrame(ohclv_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
	smoothened_ohclv_data = smoother.PandaSmoother(ohclv_df)
	return  smoothened_ohclv_data, np.array(data["Close"]), list(data["Date"])


def getTechnicalIndicators(X,d):

	RSI = getRSI(X[:,3]) 
	StochasticOscillator = getStochasticOscillator(X)
	Williams = getWilliams(X)
	MACD = getMACD(X[:,3])
	PROC = getPriceRateOfChange(X[:,3],d)
	OBV = getOnBalanceVolume(X)

	min_len = min(len(RSI),
				  len(StochasticOscillator),
				  len(Williams),
				  len(MACD),
				  len(PROC),
				  len(OBV))

	RSI = RSI[len(RSI) - min_len:]
	StochasticOscillator = StochasticOscillator[len(StochasticOscillator) - min_len:]
	Williams = Williams[len(Williams) - min_len: ]
	MACD = MACD[len(MACD) - min_len:]
	PROC = PROC[len(PROC) - min_len:]
	OBV = OBV[len(OBV) - min_len:]

	
	feature_matrix = np.c_[RSI[:,0],
						   StochasticOscillator[:,0],
						   Williams[:,0],
						   MACD[:,0],
						   PROC[:,0],
						   OBV[:,0]]

	return feature_matrix

def prepareData(X, close, date, d, plot_days=None):
    day = int(d)
    feature_matrix = getTechnicalIndicators(X, day)
    
    number_of_samples = np.shape(feature_matrix)[0]
    date = date[len(date) - number_of_samples:]
    close = close[len(close) - number_of_samples:]

    y0 = close[:number_of_samples - day]
    y1 = close[day:]

    feature_matrix_1 = feature_matrix[:number_of_samples - day]
    
    # Modify this line to include more days
    if plot_days is None:
        plot_days = number_of_samples  # Default to all available data if not specified
    feature_matrix_2 = feature_matrix[number_of_samples - plot_days:]
    date = date[number_of_samples - plot_days:]
    closeplot = close[number_of_samples - plot_days:]

    y = np.sign(y1 - y0)
    feature_matrix_1 = feature_matrix_1[:, range(6)]

    return feature_matrix_1, y, feature_matrix_2[:, range(6)], closeplot, date


	

def plotTradingStrategy(model, xplot, closeplot, Trading_Day,date):

	colorMap = {-1.0:"r",1.0:"b",0.0:"y"}
	tradeMap = {-1.0:"Sell",1.0:"Buy",0.0:"Buy"}
	plt.figure()
	plt.plot(closeplot, c = "g")
	x = [xplot[i] for i in range(0,len(xplot),Trading_Day)]
	y = [closeplot[i] for i in range(0, len(closeplot),Trading_Day)]
	y_pred = model.predict(x)
   
	c = [colorMap[y_pred[i]] for i in range(len(y_pred))]

	df = pd.DataFrame(np.c_[[ i+1 for i in range(0, len(xplot),Trading_Day)], x, y, [tradeMap[y_pred[i]] for i in range(len(y_pred)) ]],
			columns = ["Day","RSI","Stochastic Oscillator","Williams","MACD","Price Rate Of Change","On Balance Volume","Close","Buy/Sell"])
	print(df)
	df.to_csv("AAPLBuySellTradePoints.csv",index = False)


	plt.scatter([i for i in range(0,len(xplot),Trading_Day)],y, c = c)
	plt.xticks([i for i in range(0,len(xplot),Trading_Day)],[date[i] for i in range(0,len(xplot),Trading_Day)])
	red_patch = mpatches.Patch(color='red', label='Sell')
	blue_patch = mpatches.Patch(color = "blue", label = "Buy")
	plt.legend(handles = [red_patch,blue_patch])
	plt.xlabel("Time")
	plt.ylabel("Closing price")
	plt.title("Trading strategy for {} days trading window".format(Trading_Day))
	plt.savefig("TradingStrategy.png")
	plt.show(block = False)

def main(stock_symbol,Trading_Day, plot_days=None):

	fetcher = DataFetcher()
	try:
		data = fetcher.getHistoricalData(stock_symbol)
	except FileNotFoundError as e:
		print(e)
		return

	ohclv_data, close, date = getData(data)
	ohclv_data = np.array(ohclv_data)

	X,y,xplot,closeplot,dateplot = prepareData(ohclv_data, close, date, Trading_Day, plot_days)
	y[y == 0] = 1

	Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, random_state = 0)

	#model = RandomForestClassifier(n_estimators = 200,bootstrap=False, max_features="sqrt", min_samples_leaf=1, min_samples_split=2, criterion="gini")
	model=RandomForestClassifier()
	scores = cross_val_score(model, Xtrain, ytrain, cv = 5)	
	print (set(ytrain))
	print ("Cross Validation scores")
	for i, score in enumerate(scores):
		print ("Validation Set {} score: {}".format(i, score))
	model.fit(Xtrain, ytrain)
	y_pred = model.predict(Xtest)
	Eval = Evaluator(Xtest,ytest,y_pred,model)
	
	accuracy, recall, precision, specificity = Eval.getPerformanceMetrics()
	
	print ("")
	print ("Accuracy:",accuracy)
	print ("Recall:",recall)
	print ("Precision:",precision)
	print ("Specificity:",specificity)

	Eval.plotClassificationResult()
	Eval.drawROC()
	plotTradingStrategy(model,xplot,closeplot,Trading_Day,dateplot)
	Eval.printActualVsPredicted()

	c = input("Press y to generate OOB vs Number of estimators graph:")
	if c == "y" or c == "Y":
		Eval.oob_vs_n_trees(100,Xtrain,ytrain) 




	# raw_input("Press enter to genereate OOB vs Number of estimators graph:")
	# p.start()
	# print "LOL"
	# p.join()

	

stock_symbol = input("Enter the stock_symbol (AAPL, AMZN, MSFT): ")
Trading_Day = input("Enter the trading window: ")
Trading_Day_int = int(Trading_Day)
plot_days = int(input("Enter the number of days to plot: "))
main(stock_symbol.upper(),Trading_Day_int, plot_days)