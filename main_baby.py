# Keystroke Dynamics Analysis and Prediction

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras
from tensorflow.keras import layers

import csv

from plots import *
from EER import *

#TABS NOT SPACES!!!!

df = pd.read_csv("./dataset/DSL-StrongPasswordData.csv") #open csv file for reading in pandas dataframe

print(df.info()) #Get info about dataframe
print(df) #look at first handful of entries to know what we are dealing with


df_original = df.copy() #Always keep an original copy of the dataframe before any changes are made to it, just in case it is useful in the future!

#Lets plot a few things to understand what the data looks like
### plot some things!

'''
def swarmplot(y_var, x_var, dataframe):
	sns_plot = sns.swarmplot(y=df[y_var], x=df[x_var], data=dataframe, s=1, size = 20)
	fig1 = sns_plot.get_figure()
	sns_plot.set_xticklabels(sns_plot.get_xticklabels(),rotation = 90)
	sns_plot.tick_params(axis='x', labelsize=5)
	fig1.savefig('plots/' + y_var + '_' + x_var + '.png', dpi = 300)

	sns_boxplot = sns.boxplot(y=df[y_var], x=df[x_var], data=dataframe, whis=np.inf)
	sns_boxplot.set_xticklabels(sns_plot.get_xticklabels(),rotation = 90)
	sns_boxplot.tick_params(axis='x', labelsize=5)
	fig2 = sns_boxplot.get_figure()
	fig2.savefig('plots/' + y_var + '_' + x_var + '_boxplot.png', dpi = 300)

#DO NOT UNCOMMENT THE BELOW! IT HAS ALREADY BEEN RUN AND UNCOMMENTING THIS WILL WASTE HOURS OF YOUR LIFE!!!!

for column in df.columns[3:]:
	swarmplot(str(column), 'subject', df)
	
#Scatterplot of PPD vs RPD  PPD = H + UD  RPD = UD  password = .tie5Roanl
def scatterplot(y_var, x_var, dataframe):
	y_label = y_var
	x_label = x_var + " + " + y_var
	x_label_save = x_var + "+" + y_var
	sns_scatterplot = sns.scatterplot(y=df[y_var], x=( df[x_var] + df[y_var] ), hue = 'subject', data=df, s=2)
	sns_scatterplot.set_xlabel(x_label)
	sns_scatterplot.set_ylabel(y_label)
	handles, labels = sns_scatterplot.get_legend_handles_labels()
	sns_scatterplot.legend(handles[:dataframe.subject.nunique()], labels[:dataframe.subject.nunique()])
	sns_scatterplot.legend(loc=4, prop={'size': 1})  #NEED TO MAKE LEGEND SMALLER!!!!
	sns_scatterplot.legend()
	fig3 = sns_scatterplot.get_figure()
	fig3.savefig('plots/' + y_label + '_' + x_label_save + '_scatterplot.png', dpi = 300)

strings = ['period', 't', 'i', 'e', 'five', 'Shift.r', 'o', 'a', 'n', 'l']

for i in range(len(strings) - 1):
	string_first = strings[i]
	string_second = strings[i+1]
	scatterplot('UD.' + string_first + '.' + string_second, 'H.' + string_second, df)
'''




#As the feature 'subject' is an object in the dataset, we need to convert it to something that can be trained on.
subjects = df['subject'].unique()  #Store all unique subject IDs
subjects_to_int = {subject: i  for i, subject in enumerate(subjects)} #Change all unique subject IDs(key) to integers(value) and store in dictionary
int_to_subjects = {i: subject for i, subject in enumerate(subjects)} #..and vice versa! integers(key) and original subject IDs(values)

df = df.replace(subjects_to_int) #replace subject column with subjects_to_int
print(df.head)
print(df.info() )


def df_to_dataset(dataframe):  #Function to convert the dataframe to a tensorflow dataframe that can be ML'd on. Temporarily remove the target column
	dataframe = dataframe.copy()
	new_df = dataframe
	new_df =tf.convert_to_tensor(new_df)
	return new_df

unique_users = df['subject'].nunique()


def df_to_dataset(dataframe,  batch_size=1):  #Function to convert the dataframe to a tensorflow dataframe that can be ML'd on
	dataframe = dataframe.copy()
	new_df =tf.convert_to_tensor(dataframe)
	return new_df 

def nn_model(input_dim, output_dim=1, nodes=31):
	model = keras.Sequential()
	model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
	model.add(Dense(nodes, activation='relu'))
	model.add(Dense(output_dim, activation='sigmoid'))
	optimiser = keras.optimizers.Adam(learning_rate = 0.0001)
	model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])
	return model



	
eer_per_user_dict = {}

def evaluateEER(user_scores, imposter_scores, subject):
	labels = [0]*len(user_scores) + [1]*len(imposter_scores)
	fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
	roc_auc = metrics.auc(fpr, tpr)
	missrates = 1 - tpr
	farates = fpr
	dists = missrates - farates
	idx1 = np.argmin(dists[dists >= 0])
	idx2 = np.argmax(dists[dists < 0])
	x = [missrates[idx1], farates[idx1]]
	y = [missrates[idx2], farates[idx2]]
	a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
	eer = x[0] + a * ( y[0] - x[0] )
	print('subject: ', subject ,' eer: ', eer)
	eer_per_user_dict[subject] = eer
	return eer

def ROCplot (subject, user_scores, imposter_scores):
	labels = [0]*len(user_scores) + [1]*len(imposter_scores)
	fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
	roc_auc = metrics.auc(fpr, tpr)
	plt.figure() 
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig('plots/ROC_' + str(subject) + '.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()



def plot_loss(x , y, subject):
	plt.plot(x)  #Make a plot of the mae (similar to loss) vs. epochs
	plt.plot(y)
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.show()
	plt.savefig('plots/ROC_' + str(subject) + '.png')

#Create dictionaries to hold multiple dataframes
df_temp_dict = {}
df_train_dict = {}
df_test_dict = {}  #used as positive examples
df_imposter_dict = {} #used as negative examples
scaler = MinMaxScaler()  #Need to scale between 0-1 as using ReLu activation function
eers = []

for subject in range(unique_users):

	user_scores = []
	imposter_scores = []

	df_temp_dict[subject] = pd.DataFrame()
	df_train_dict[subject] = pd.DataFrame()
	df_test_dict[subject] = pd.DataFrame()
	df_imposter_dict[subject] = pd.DataFrame()
	grouped = df.groupby(['subject'])
	df_temp_dict[subject] = grouped.get_group(subject)
	df_train_dict[subject] = df_temp_dict[subject].sample(n=200)  # Instead of taking first 200 I am sampling randomly
	df_test_dict[subject] = df_temp_dict[subject].drop(df_train_dict[subject].index) 
	imposter_data = df.loc[df.subject != subject, :]  
	df_imposter_dict[subject] = imposter_data.groupby("subject").head(5)


	df_train_dict[subject] = pd.DataFrame(scaler.fit_transform(df_train_dict[subject]), columns=df_temp_dict[subject].columns)
	df_test_dict[subject] = pd.DataFrame(scaler.fit_transform(df_test_dict[subject]), columns=df_temp_dict[subject].columns)
	df_imposter_dict[subject] = pd.DataFrame(scaler.fit_transform(df_imposter_dict[subject]), columns=df_temp_dict[subject].columns)

	to_drop = ['subject', 'sessionIndex', 'rep' ]  #List of columns of data to drop from dataframe as they are irrelevant for training
	df_train_dict[subject].drop(to_drop, inplace=True, axis=1) #Remove the columns of data
	df_test_dict[subject].drop(to_drop, inplace=True, axis=1) #Remove the columns of data
	df_imposter_dict[subject].drop(to_drop, inplace=True, axis=1) #Remove the columns of data

	train_ds = df_to_dataset(df_train_dict[subject])
	test_ds = df_to_dataset(df_test_dict[subject])
	imposter_ds = df_to_dataset(df_imposter_dict[subject])
	# One hot encoding of target vector
	Y = pd.get_dummies(df_test_dict[subject]).values
	n_classes = Y.shape[1]


  	# Train the neural network model
	n_features = df_train_dict[subject].shape[1]
	model = nn_model(n_features, n_classes, 31)
	history = model.fit(train_ds, test_ds, epochs=3, batch_size=5)  #test with 3 epochs, but use 300 otherwise

	# Predict on the NN
	prediction_test = model.predict(test_ds)
	for pred in prediction_test:
		user_scores.append(pred[0])
	prediction_imposter = model.predict(imposter_ds)
	for pred in prediction_imposter:
		imposter_scores.append(pred[0])
	for key, value in int_to_subjects.items():
		if key == subject:
			user_id = value
			ROCplot(user_id, user_scores, imposter_scores)
			plot_loss(history.history['loss'], history.history['accuracy'], user_id)
			eers.append(evaluateEER(user_scores, imposter_scores, user_id))
	

print('eer')
print('eer mean: ' , np.mean(eers), ' eer std: ', np.std(eers))

print(eer_per_user_dict)


def eerPlot(data):

	names = list(data.keys())
	values = list(data.values())
	plt.figure()
	plt.bar(range(len(data)), values, tick_label=names)
	plt.title('EER per user')
	plt.xlabel('user', fontsize = 1)
	plt.xticks(rotation=90)
	plt.ylabel('EER')
	plt.draw()
	plt.savefig('plots/EER_per_user.png')

eerPlot(eer_per_user_dict)



	




