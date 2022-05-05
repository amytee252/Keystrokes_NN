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

def metrics(subject, user, history):

	if subject is 0:
		fnr_name = 'false_negatives' 
		fpr_name = 'false_positives' 
		tpr_name = 'true_positives' 
		tnr_name = 'true_negatives' 
	else:
		fnr_name = 'false_negatives_' + str(subject)
		fpr_name = 'false_positives_' + str(subject)
		tpr_name = 'true_positives_' + str(subject)
		tnr_name = 'true_negatives_' + str(subject)

	fnr = history.history[fnr_name][-1]
	fpr = history.history[fpr_name][-1]
	tpr = history.history[tpr_name][-1]
	tnr = history.history[fnr_name][-1]

	if fnr + tpr is 0.0:
		print('cannot calculate recall, precision, F1 score')
	elif tpr is 0.0:
		print('cannot calculate recall, precision, F1 score')
		
	else:
	
		recall = tpr / (tpr + fnr)
		precision = tpr / (tpr + fpr)
		F1_score = 2 * ((precision * recall) / (precision + recall))

		print('For user: ', user )
		print('Recall: ', recall)
		print('Precision: ', precision)
		print('F1 score: ', F1_score)
	

