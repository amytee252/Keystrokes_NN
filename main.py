# Keystroke Dynamics Analysis and Prediction

import numpy as np
import pandas as pd

import matplotlib as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

#TABS NOT SPACES!!!!

df = pd.read_csv("./dataset/DSL-StrongPasswordData.csv") #open csv file for reading in pandas dataframe

print(df.info()) #Get info about dataframe
print(df) #look at first handful of entries to know what we are dealing with

df_original = df.copy() #Always keep an original copy of the dataframe before any changes are made to it, just in case it is useful in the future!

#Lets plot a few things to understand what the data looks like










#As the feature 'subject' is an object in the dataset, we need to convert it to something that can be trained on.
subjects = df['subject'].unique()  #Store all unique subject IDs
subjects_to_int = {subject: i  for i, subject in enumerate(subjects)} #Change all unique subject IDs(key) to integers(value) and store in dictionary
int_to_subjects = {i: subject for i, subject in enumerate(subjects)} #..and vice versa! integers(key) and original subject IDs(values)

df = df.replace(subjects_to_int) #replace subject column with subjects_to_int
print(df.head)
print(df.info() )

unique_users = df['subject'].nunique()


def df_to_dataset(dataframe,  batch_size=1):  #Function to convert the dataframe to a tensorflow dataframe that can be ML'd on
	dataframe = dataframe.copy()
	new_df =tf.convert_to_tensor(dataframe)
	return new_df 

batch_size = 1  #adjust

#Create dictionaries to hold multiple dataframes
df_temp_dict = {}
df_train_dict = {}
df_test_dict = {}  #used as positive examples
df_imposter_dict = {} #used as negative examples
scaler = MinMaxScaler()  #Need to scale between 0-1 as using ReLu activation function
for subject in range(unique_users):
	df_temp_dict[subject] = pd.DataFrame()
	df_train_dict[subject] = pd.DataFrame()
	df_test_dict[subject] = pd.DataFrame()
	df_imposter_dict[subject] = pd.DataFrame()
	grouped = df.groupby(['subject'])
	df_temp_dict[subject] = grouped.get_group(subject)
	df_train_dict[subject] = df_temp_dict[subject].sample(n=200)  #change to first 200
	df_test_dict[subject] = df_temp_dict[subject].drop(df_train_dict[subject].index) #Change to last 200
	imposter_data = df.loc[df.subject != subject, :]  
	df_imposter_dict[subject] = imposter_data.groupby("subject").head(5)#.loc[:, "H.period":"H.Return"]
	if subject is  50:
		print("Last user's dataframe")
		print(df_imposter_dict[subject])
	#df_temp_dict[subject] = pd.DataFrame(scaler.fit_transform(df_temp_dict[subject]), columns=df_temp_dict[subject].columns)  #Normalizes the data in the dataframe
	df_train_dict[subject] = pd.DataFrame(scaler.fit_transform(df_train_dict[subject]), columns=df_temp_dict[subject].columns)
	df_test_dict[subject] = pd.DataFrame(scaler.fit_transform(df_test_dict[subject]), columns=df_temp_dict[subject].columns)
	df_imposter_dict[subject] = pd.DataFrame(scaler.fit_transform(df_imposter_dict[subject]), columns=df_temp_dict[subject].columns)
	#print(df_train_dict[subject].info)
	#print(df_test_dict[subject].info)
	if subject is  50:
		print("Last user's dataframe with scaling applied")
		print(df_imposter_dict[subject])
	to_drop = ['subject', 'sessionIndex', 'rep' ]  #List of columns of data to drop from dataframe as they are irrelevant for training
	df_train_dict[subject].drop(to_drop, inplace=True, axis=1) #Remove the columns of data
	df_test_dict[subject].drop(to_drop, inplace=True, axis=1) #Remove the columns of data
	df_imposter_dict[subject].drop(to_drop, inplace=True, axis=1) #Remove the columns of data
	if subject is  50:
		print("Last user's dataframe when columns are dropped")
		print(df_imposter_dict[subject].info())
		print(df_imposter_dict[subject])
	#test_ds_[subject] = df_to_dataset(df_test_dict[subject],  batch_size=batch_size)
	#imposter_ds_[subject] = df_to_dataset(df_imposter_dict[subject],  batch_size=batch_size)

#df_train = pd.concat(df_train_dict.values(), ignore_index=True)
#print(df_train)
#df_test = pd.concat(df_test_dict.values(), ignore_index=True)
#print(df_test)


model = tf.keras.Sequential([  #build NN
			tf.keras.layers.Dense(31, activation='relu'),   #Change it to get 31 from length of columns
			tf.keras.layers.Dense(8 , activation='relu'),
			tf.keras.layers.Dense(1 , activation='sigmoid')
])



#Train is train
#test is test and again test with imposter
