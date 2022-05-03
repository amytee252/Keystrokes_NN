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



eer_per_user_dict = {}  # Dictionary to hold the eer values per user calculated in function below

# Function to calculate the EER
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
