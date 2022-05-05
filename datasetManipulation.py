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


def scaling(dataframe):
	dataframe = dataframe.copy()
	new_df = dataframe
	scaler = MinMaxScaler() #Need to scale as using ReLu
	standardised_features = pd.DataFrame(scaler.fit_transform(new_df[features].copy()), columns=features) 
	#print(standardised_features)
	old_shape = new_df.shape
	new_df.drop(features, axis = 1, inplace = True)
	# join back the normalized features
	new_df = pd.concat([new_df, standardised_features], axis= 1)
	assert old_shape == new_df.shape, "something went wrong!"
	return new_df



def datasetTransformations(dataframe):
	dataframe = dataframe.copy()
	new_df = dataframe
	to_drop = ['subject', 'sessionIndex', 'rep' ]  #List of columns of data to drop from dataframe as they are irrelevant for training
	new_df.drop(to_drop, inplace=True, axis=1)
	return new_df


def df_to_dataset(dataframe):  #Function to convert the dataframe to a tensorflow dataframe that can be ML'd on. Temporarily remove the target column
	dataframe = dataframe.copy()
	new_df = dataframe
	new_df =tf.convert_to_tensor(new_df)
	return new_df
