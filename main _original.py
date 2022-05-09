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
from model import *
from datasetManipulation import *
from features import *
from metrics import *



df = pd.read_csv("./dataset/DSL-StrongPasswordData.csv") #open csv file for reading in pandas dataframe

print(df.info()) #Get info about dataframe
print(df) #look at first handful of entries to know what we are dealing with


#Lets plot a few things to understand what the data looks like
#DO NOT UNCOMMENT THE BELOW! IT HAS ALREADY BEEN RUN AND UNCOMMENTING THIS WILL WASTE HOURS OF YOUR LIFE TRYING TO PLOT STUFF THAT ALREADY EXISTS !!!!

#for column in df.columns[3:]:
#	swarm_plot(str(column), 'subject', df)
	
#strings = ['period', 't', 'i', 'e', 'five', 'Shift.r', 'o', 'a', 'n', 'l']

#for i in range(len(strings) - 1):
#	string_first = strings[i]
#	string_second = strings[i+1]
#	scatter_plot('UD.' + string_first + '.' + string_second, 'H.' + string_second, df)





#As the feature 'subject' is an object in the dataset, we need to convert it to something that can be trained on.
subjects = df['subject'].unique()  #Store all unique subject IDs
subjects_to_int = {subject: i  for i, subject in enumerate(subjects)} #Change all unique subject IDs(key) to integers(value) and store in dictionary
int_to_subjects = {i: subject for i, subject in enumerate(subjects)} #..and vice versa! integers(key) and original subject IDs(values)

df = df.replace(subjects_to_int) #replace subject column with subjects_to_int
print(df)
print(df.info() )


unique_users = df['subject'].nunique()

#Create dictionaries to hold multiple dataframes
df_temp_dict = {}
df_train_dict = {}
df_test_dict = {}  #used as positive examples
df_imposter_dict = {} #used as negative examples

eers = []


df = scaling(df)
print(df)

grouped = df.groupby(['subject'])

for subject in range(unique_users):

	user_scores = []
	imposter_scores = []

	df_temp_dict[subject]  = pd.DataFrame()
	df_train_dict[subject] = pd.DataFrame()
	df_test_dict[subject] = pd.DataFrame()
	df_imposter_dict[subject] = pd.DataFrame()
	

	df_temp_dict[subject] = grouped.get_group(subject)

	df_train_dict[subject] = df_temp_dict[subject].sample(n=200)  # Instead of taking first 200 I am sampling randomly
	df_test_dict[subject] = df_temp_dict[subject].drop(df_train_dict[subject].index)

	imposter_data = df.loc[df.subject != subject, :]  
	df_imposter_dict[subject] = imposter_data.groupby("subject").head(5)
	

	df_train_dict[subject] = datasetTransformations(  df_train_dict[subject])
	df_test_dict[subject] = datasetTransformations(  df_test_dict[subject]) 
	df_imposter_dict[subject] = datasetTransformations(  df_imposter_dict[subject]) 


	Y = pd.get_dummies(df_train_dict[subject]).values
	n_classes = Y.shape[1] 
	print('n_classes: ', n_classes)

  	# Train the neural network model
	n_features = df_train_dict[subject].shape[1]
	print('n_features: ', n_features)
	model = nn_model(n_features, 1, 31)
	history = model.fit(np.array(df_train_dict[subject]), np.ones(df_train_dict[subject].shape[0]), epochs=100, batch_size=5)  
	# NOTE: WE are designating normal (user) with the label 1! An imposter would have a prediction closer to 0. However, normally users are labeled with 0 and imposters with 1

	print(history.history.keys())

	# Predict on the NN
	prediction_test = 1.0 - model.predict(np.array(df_test_dict[subject]))
	for pred in prediction_test:
		user_scores.append(pred[0])
	prediction_imposter = 1.0 - model.predict(np.array(df_imposter_dict[subject]))
	for pred in prediction_imposter:
		imposter_scores.append(pred[0])

	for key, value in int_to_subjects.items():
		if key == subject:
			user_id = value
			ROC_plot(user_id, user_scores, imposter_scores)
			EER_plot(user_id, user_scores, imposter_scores)
			loss_plot(history.history['loss'], history.history['accuracy'], user_id) #x, y
			eers.append(evaluateEER(user_scores, imposter_scores, user_id))
			metrics(subject, user_id, history)
	K.clear_session() #clears the current model Need this to not break 'metrics'


	

print('eer')
print('eer mean: ' , np.mean(eers), ' eer std: ', np.std(eers))

print(eer_per_user_dict)

EER_bar_plot(eer_per_user_dict)




	




