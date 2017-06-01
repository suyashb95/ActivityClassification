'''
Script to make predictions on raw readings streaming
over a UDP socket
'''


import pandas as pd
import os, pickle, socket
import numpy as np
import scipy.stats as stats
from statsmodels.robust.scale import mad as mediad
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sb
from matplotlib import pyplot as plt
import itertools
from sklearn import preprocessing
from collections import deque

UDPSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDPSocket.bind(('0.0.0.0', 5000))


X = deque(maxlen=400)
Y = deque(maxlen=400)
Z = deque(maxlen=400)

NBModel = pickle.load(open('NBModel2.pkl', 'rb'))

def get_parameters(data):
	'''
	perform aggregates on 
	rolling windows of 410
	With scaling and normalizing options
	'''
	func_dict = {
		'min': np.min,
		'max': np.max,
		'diff': lambda x: np.max(x) - np.min(x),
		'std': np.std,
		'iqr': stats.iqr,
		'rms': lambda x: np.sqrt(np.mean(np.square(x))),
		'mad': lambda x: x.mad(),
		'mediad': mediad
	}
	aggregations = {
		'X': func_dict,
		'Y': func_dict,
		'Z': func_dict
	}
	#Preprocessing Options
	#data[['X', 'Y', 'Z']] = preprocessing.scale(preprocessing.MinMaxScaler().fit_transform(data[['X', 'Y', 'Z']]))
	#data[['X', 'Y', 'Z']] = preprocessing.scale(data[['X', 'Y', 'Z']])
	data['temp'] = 10
	data = data.groupby('temp', as_index=False)
	stats_data = data.agg(aggregations)
	stats_data.columns = [''.join(col).strip() for col in stats_data.columns.values]
	del stats_data['temp']
	return stats_data

def predict():
	count = 0
	while True:
		data, _ = UDPSocket.recvfrom(4096)
		readings = str(data, 'utf-8').split(',')
		X.append(float(readings[0]))
		Y.append(float(readings[1]))
		Z.append(float(readings[2]))
		count += 1
		if count == 200:
			data_frame = pd.DataFrame({
					'X': X,
					'Y': Y,
					'Z': Z,
				}
			)
			print(NBModel.predict(get_parameters(data_frame)))
			count = 0


if __name__ == '__main__':
	predict()


