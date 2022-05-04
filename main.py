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



df = pd.read_csv("./dataset/DSL-StrongPasswordData.csv") #open csv file for reading in pandas dataframe

print(df.info()) #Get info about dataframe
print(df) #look at first handful of entries to know what we are dealing with


df_original = df.copy() #Always keep an original copy of the dataframe before any changes are made to it, just in case it is useful in the future!

#Lets plot a few things to understand what the data looks like

#DO NOT UNCOMMENT THE BELOW! IT HAS ALREADY BEEN RUN AND UNCOMMENTING THIS WILL WASTE HOURS OF YOUR LIFE TRYING TO PLOT STUFF THAT ALREADY EXISTS !!!!

#for column in df.columns[3:]:
#	swarmplot(str(column), 'subject', df)
	
#strings = ['period', 't', 'i', 'e', 'five', 'Shift.r', 'o', 'a', 'n', 'l']

#for i in range(len(strings) - 1):
#	string_first = strings[i]
#	string_second = strings[i+1]
#	scatterplot('UD.' + string_first + '.' + string_second, 'H.' + string_second, df)





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

def nn_model(input_dim, output_dim=1, nodes=31):
	model = keras.Sequential()
	model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
	model.add(Dense(nodes, activation='relu'))
	model.add(Dense(output_dim, activation='sigmoid'))
	optimiser = keras.optimizers.Adam(learning_rate = 0.0001) #default parameters used except for lr.
	model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
	return model




#Create dictionaries to hold multiple dataframes
df_temp_dict = {}
df_train_dict = {}
df_test_dict = {}  #used as positive examples
df_imposter_dict = {} #used as negative examples
scaler = MinMaxScaler()  #Need to scale between 0-1 as using ReLu activation function
eers = []

#MAKE A FUNCTION WHICH DOES THIS DATAFRAME MANIPULATION AND THEN PASS IT IN

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
	history = model.fit(train_ds, Y, epochs=500, batch_size=5)  #test with 3 epochs, but use 500 otherwise

	# Predict on the NN
	prediction_test = 1.0 - model.predict(test_ds)
	for pred in prediction_test:
		user_scores.append(pred[0])
	prediction_imposter = 1.0 - model.predict(imposter_ds)
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

eerPlot(eer_per_user_dict)

#TO DO:
# F1 score, precision, recall
# Plot FRR and FAR


	




