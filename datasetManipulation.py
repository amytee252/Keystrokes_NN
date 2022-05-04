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

def datasetTransformations(dataframe):
	dataframe = dataframe.copy()
	new_df = dataframe
	scaler = MinMaxScaler() #Need to scale as using ReLu
	new_df = pd.DataFrame(scaler.fit_transform(new_df), columns=new_df.columns)
	to_drop = ['subject', 'sessionIndex', 'rep' ]  #List of columns of data to drop from dataframe as they are irrelevant for training
	new_df.drop(to_drop, inplace=True, axis=1)
	return new_df


def df_to_dataset(dataframe):  #Function to convert the dataframe to a tensorflow dataframe that can be ML'd on. Temporarily remove the target column
	dataframe = dataframe.copy()
	new_df = dataframe
	new_df =tf.convert_to_tensor(new_df)
	return new_df
