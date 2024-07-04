import numpy as np
import pandas as pd

class DataPreprocessor:

	def __init__(self):
		pass

	def CustomSmoother(self,x,alpha):

		s0 = x[0]
		smoothed_statistic = [s0]
		n = x.shape[0] 
		for i in range(1,n):
			s1 = alpha * x[i] + (1 - alpha) * s0
			smoothed_statistic.append(s1)
			s0 = s1
		smoothed_statistic = np.array(smoothed_statistic)
		return smoothed_statistic

	def PandaSmoother(self, x):

		return x.ewm(span=20).mean()
