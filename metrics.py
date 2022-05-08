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
from keras import backend as K 

import csv

from plots import *
from EER import *
from model import *
from datasetManipulation import *
from features import *


#MASSIVE NOTE: I need to check the if statements catch all cases... they may not.
def metrics(subject, user, history):

	fnr_name = 'false_negatives_1'
	fpr_name = 'false_positives_1'
	tpr_name = 'true_positives_1' 
	tnr_name = 'true_negatives_1'

	fnr = history.history[fnr_name][-1]  #Returns last element in sequence
	fpr = history.history[fpr_name][-1]
	tpr = history.history[tpr_name][-1]
	tnr = history.history[fnr_name][-1]

	if tpr == 0.0:
		recall = 0.0
		precision = 0.0
		print('cannot calculate recall or precision')
	else:

		recall = (tpr / (tpr + fnr))
		precision = (tpr / (tpr + fpr))

	if (recall == 0.0 or precision == 0.0):
		F1_score = 0.0
		print('cannot calculate F1 score')
	else:
		F1_score = 2 * ((precision * recall) / (precision + recall))

		print('For user: ', user )
		print('Recall: ', recall)
		print('Precision: ', precision)
		print('F1 score: ', F1_score)
	

