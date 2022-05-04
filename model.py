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



def nn_model(input_dim, output_dim=1, nodes=31):
	model = keras.Sequential()
	model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
	model.add(Dropout(0.02))
	model.add(Dense(nodes, activation='relu'))
	model.add(Dropout(0.02))
	model.add(Dense(nodes, activation='relu'))
	model.add(Dropout(0.02))
	model.add(Dense(output_dim, activation='sigmoid'))
	optimiser = keras.optimizers.Adam(learning_rate = 0.0001) #default parameters used except for lr.
	model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])
	return model
