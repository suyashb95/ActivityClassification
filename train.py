'''
Script for data pre-processing and training
'''

import pandas as pd
import os, pickle
import numpy as np
import scipy.stats as stats
from statsmodels.robust.scale import mad as mediad
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sb
from matplotlib import pyplot as plt
import itertools
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

def read_raw_data():
	'''
	Read all the text files in the datasets(diliac) folder
	concatenate them into one dataframe
	'''
	data_frames = []
	current_dir = os.getcwd()
	os.chdir('../datasets')
	for item in os.listdir(os.getcwd()):
		if os.path.isfile(item):
			if item.endswith('.txt'):
				data_frames.append(pd.read_csv(item, header=None, usecols=[0, 1, 2, 24]))
	os.chdir(current_dir)
	data_frame = pd.concat(data_frames)
	data_frame.columns = ['X', 'Y', 'Z', 'Activity']
	return data_frame

def split_activities(data_frame):
	'''
	split accelerometer readings according to activities
	combine some classes 
	return three dataframes
	'''
	activities_list = [
		[1, 2, 3],
		[7, 8, 9],
		[10],
		[11, 12]
	]
	df_list = []
	for (index, activities) in enumerate(activities_list):
		df = data_frame[data_frame['Activity'].isin(activities)]
		df['Activity'] = index + 1
		df_list.append(df)
	return tuple(df_list)

def preprocess_custom_data():
	data = pd.read_csv('merged_log.txt', header=None)
	data.columns = ['X', 'Y', 'Z', 'Activity']
	stationary = data[data['Activity'].isin([0])]
	walking = data[data['Activity'].isin([1])]
	running = data[data['Activity'].isin([2])]
	stats_stationary = get_parameters(stationary)
	stats_walking = get_parameters(walking, chunk_size=200)
	stats_running = get_parameters(running, chunk_size=200)
	pd.concat([stats_stationary, stats_walking, stats_running]).to_csv('custom_dataset_params4.csv')

def get_parameters(data, chunk_size=410):
	'''
	perform aggregates on 
	rolling windows of 410
	after scaling and normalizing
	This function seems to be faster,
	no unnecessary calculations
	'''
	activity = data['Activity']
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
	data_groups = []
	for i in range(int(data.shape[0]/(chunk_size/2)) - 1):
		temp = data.iloc[int(i*(chunk_size/2)):int((i+2)*(chunk_size/2))]
		temp['k'] = i
		data_groups.append(temp)
	data_groups = pd.concat(data_groups).groupby('k', as_index=False)
	stats_data = data_groups.agg(aggregations)
	stats_data.columns = [''.join(col).strip() for col in stats_data.columns.values]
	activity = activity.reset_index(drop=True)
	stats_data = pd.concat([stats_data, activity[:len(stats_data)]], axis=1)
	del stats_data['k']
	return stats_data

def get_parameters_slow(data, chunk_size=410):
	'''
	perform aggregates on 
	rolling windows of 410
	'''
	activity = data['Activity']
	func_dict = {
		'min': np.min,
		'max': np.max,
		'diff': lambda x: np.max(x) - np.min(x),
		'mean': np.mean,
		'std': np.std,
		'iqr': stats.iqr,
		'rms': lambda x: np.sqrt(np.mean(np.square(x))),
		'integral': np.trapz,
		'mad': lambda x: np.fabs(x - x.mean()).mean(),
		'mediad': mediad
	}
	aggregations = {
		'X': func_dict,
		'Y': func_dict,
		'Z': func_dict
	}
	data_groups = data.rolling(window=chunk_size)
	stats_data = data_groups \
		.agg(aggregations) \
		.iloc[chunk_size-1::chunk_size/2] \
		.reset_index(drop=True)
	correlations = data_groups[['X', 'Y', 'Z']] \
		.corr() \
		.iloc[chunk_size-1::chunk_size/2] \
		.to_frame() \
		.transpose() \
		.reset_index(drop=True)
	correlations.columns = correlations.columns.droplevel()
	correlations.columns = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
	correlations = correlations.drop(['XX', 'YY', 'ZZ', 'YX', 'XZ', 'ZY'], axis=1)
	stats_data.columns = [''.join(col).strip() for col in stats_data.columns.values]
	activity = activity.reset_index(drop=True)
	stats_data = pd.concat([stats_data, correlations, activity[:len(stats_data)]], axis=1)
	return stats_data

def save_parameters(filename='diliac_set7.csv'):
	raw_data = read_raw_data()
	print("Raw data read complete")
	training_data = []
	for activity in split_activities(raw_data):
		training_data.append(get_parameters(activity))
	training_data = pd.concat(training_data)
	training_data.to_csv(filename)

def testNB():
	data = pd.read_csv('walking_log_new.txt', header=None)
	data.columns = ['X', 'Y', 'Z', 'Activity']
	data = data[:5000]
	Y = np.array(data['Activity'])
	params = get_parameters(data)
	params = params.drop(['Activity'], axis=1)
	X = np.array(params)
	NBLearner = pickle.load(open('NBModel.pkl', 'rb'))
	accuracy = NBLearner.predict(X)
	print(accuracy)

def test_NB(model='NBModel2.pkl', data='custom_dataset_params4.csv'):
	NBLearner = pickle.load(open(model, 'rb'))
	total_data = pd.read_csv(data, index_col=0)
	print("Data reading complete")
	total_data = total_data[total_data['Activity'].isin([0, 1, 2])]
	Y = np.array(total_data['Activity'])
	X = np.array(total_data.drop(['Activity'], axis=1))
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)	
	accuracy = [1 if x else 0 for x in NBLearner.predict(X_test) == Y_test]
	cnf_matrix = confusion_matrix(Y_test, NBLearner.predict(X_test))
	print(f1_score(Y_test, NBLearner.predict(X_test), average='macro'))
	print(cnf_matrix)
	print("Accuracy: " + str(float(sum(accuracy))/len(accuracy) * 100) + "%")
	plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2],
                      title='Confusion matrix, without normalization')	

def trainNB(filename='custom_dataset_params3.csv'):
	NBLearner = GaussianNB()
	total_data = pd.read_csv(filename, index_col=0)
	print("Data reading complete")
	total_data = total_data[total_data['Activity'].isin([0, 1, 2])]
	Y = np.array(total_data['Activity'])
	X = np.array(total_data.drop(['Activity'], axis=1))
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
	NBLearner.fit(X_train, Y_train)
	print("Training complete")
	accuracy = [1 if x else 0 for x in NBLearner.predict(X_test) == Y_test]
	cnf_matrix = confusion_matrix(Y_test, NBLearner.predict(X_test))
	print(cnf_matrix)
	print("Accuracy: " + str(float(sum(accuracy))/len(accuracy) * 100) + "%")
	plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2],
                      title='Confusion matrix, without normalization')
	pickle.dump(NBLearner, open('NBModel.pkl', 'wb'))
	return NBLearner

def train_decision_tree(filename='custom_dataset_params3.csv'):
	clf = DecisionTreeClassifier(random_state=42)
	total_data = pd.read_csv(filename, index_col=0)
	print("Data reading complete")
	total_data = total_data[total_data['Activity'].isin([0, 1, 2, 3])]
	Y = np.array(total_data['Activity'])
	X = np.array(total_data.drop(['Activity'], axis=1))
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
	clf.fit(X_train, Y_train)
	print("Training complete")
	accuracy = [1 if x else 0 for x in clf.predict(X_test) == Y_test]
	cnf_matrix = confusion_matrix(Y_test, clf.predict(X_test))
	print(cnf_matrix)
	print("Accuracy: " + str(float(sum(accuracy))/len(accuracy) * 100) + "%")
	plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2, 3],
                      title='Confusion matrix, without normalization')
	pickle.dump(clf, open('DecisionTreeModel.pkl', 'wb'))
	return clf

def plot_confusion_matrix(
	cm, 
	classes,
    normalize=False,
	title='Confusion matrix',
	cmap=plt.cm.Blues
	):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_test_set(filename, y, chunk_size=200):
	'''
	Convert data logged from Serial Oscilloscope to 
	a format readable by trainNB()
	'''
	test_data = pd.read_csv(filename, header=None)
	test_data[3] = y
	test_data.columns = ['X', 'Y', 'Z', 'Activity']
	test_stats = get_parameters(test_data, chunk_size)
	test_data.to_csv(filename.split('.')[0] + '.csv')
	return test_stats

def visualise_set(filename='custom_dataset_params3.csv'):
	'''
	Plot 1000 values from each class 
	See the separation of classes 
	'''
	total_data = pd.read_csv(filename, index_col=0) \
		.groupby(['Activity']) \
		.head(500) \
		.reset_index(drop=True) \
		.select(lambda x: x[0] == 'X' or x == 'Activity', axis=1) 
	activties = total_data['Activity']
	for column in total_data:
		if column != 'Activity':
			sb.stripplot(
				x='Activity', 
				y=column, 
				data=pd.concat([total_data[column], activties], axis=1),
				jitter=True
			)
			sb.plt.show()
	#sb.pairplot(total_data, hue='Activity')
	#sb.plt.show()

if __name__ == '__main__':
	print("Starting NB")
	#test_NB()
	#trainNB()
	#train_decision_tree()
	#preprocess_custom_data()
	#test_NB()
	visualise_set()