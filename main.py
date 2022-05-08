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
print(df) #look at entries to know what we are dealing with


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


subjects = df['subject'].nunique() #number of unique subjects

#Create dictionaries to hold multiple dataframes
df_temp = {}
df_train = {}
df_test = {} 
df_imposter = {}

eers = []  #Create an array to hold the equal error rates for each user


df = scaling(df)  #Run the scaling function
print(df)

grouped = df.groupby(['subject']) #group each user in the dataframe by their username (subject)

Y = pd.get_dummies(df).values #convert categorical variable into dummy/indicator variables.
n_classes = Y.shape[1] 
print('n_classes: ', n_classes)

n_features = df.shape[1]
print('n_features: ', n_features)


class NeuralNet(keras.Sequential):
    
	def __init__(self, subjects):
		super().__init__()
		self.user_scores = []
		self.imposter_scores = []
		self.subjects = subjects
		self.learning_rate = 0.0001
		self.epochs = 100
		self.batch_size = 5
		self.inputs = 31 
		self.outputs = 1
		self.nodes = 31
		self.activation_initial = 'relu'
		self.activation_final = 'sigmoid'

	def training(self):

		self.model = keras.Sequential(
    		[
        		layers.Dense(self.inputs, activation=self.activation_initial),
       			layers.Dense(self.inputs, activation=self.activation_initial),
        		layers.Dense(self.outputs, activation = self.activation_final),
   		 ]
		)
		self.model = nn_model(self.inputs, self.outputs, self.nodes)
		self.history = self.model.fit(np.array(self.train), np.ones(self.train.shape[0]), epochs = self.epochs, batch_size = self.batch_size) 
		#print(self.model.summary() )

	def testing(self):

		prediction_test = 1.0 - self.model.predict(np.array(self.test_genuine))
		for pred in prediction_test:
			self.user_scores.append(pred[0])
		prediction_imposter = 1.0 - self.model.predict(np.array(self.test_imposter))
		for pred in prediction_imposter:
			self.imposter_scores.append(pred[0])
    
	def evaluate(self):
		eers = []


		for subject in range(subjects):
			#print(subject)
	
			self.user_scores = []
			self.imposter_scores = []

			df_temp[subject]  = pd.DataFrame()
			df_train[subject] = pd.DataFrame()
			df_test[subject] = pd.DataFrame()
			df_imposter[subject] = pd.DataFrame()
	

			df_temp[subject] = grouped.get_group(subject)

			df_train[subject] = df_temp[subject].sample(n=200)  # Instead of taking first 200 I am sampling randomly
			df_test[subject] = df_temp[subject].drop(df_train[subject].index)

			imposter_data = df.loc[df.subject != subject, :]  
			df_imposter[subject] = imposter_data.groupby("subject").head(5)
	

			self.train = datasetTransformations(  df_train[subject])
			self.test_genuine = datasetTransformations(  df_test[subject]) 
			self.test_imposter = datasetTransformations(  df_imposter[subject]) 
	
			
			self.training()
			self.testing()

			print(self.history.history.keys())
			for key, value in int_to_subjects.items():
				if key == subject:
					user_id = value
					ROC_plot(user_id, self.user_scores, self.imposter_scores)
					EER_plot(user_id, self.user_scores, self.imposter_scores)
					loss_plot(self.history.history['loss'], self.history.history['accuracy'], user_id) #x, y
					eers.append(evaluateEER(self.user_scores, self.imposter_scores, user_id))
					metrics(subject, user_id, self.history)

		return print('mean EER of all the users: ', np.mean(eers), ' std EER of all the users: ', np.std(eers))

NeuralNet(subjects).evaluate()

print(eer_per_user_dict)

EER_bar_plot(eer_per_user_dict)



	




