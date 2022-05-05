# This script contains all the plotting functions for use with the keystrokes dataset

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
# Creates swarm plots for each timing feature in the dataset
def swarm_plot(y_var, x_var, dataframe):
	sns_plot = sns.swarmplot(y=df[y_var], x=df[x_var], data=dataframe, s=1, size = 20)
	fig1 = sns_plot.get_figure()
	sns_plot.set_xticklabels(sns_plot.get_xticklabels(),rotation = 90)
	sns_plot.tick_params(axis='x', labelsize=5)
	fig1.savefig('plots/' + y_var + '_' + x_var + '.png', dpi = 300)

	sns_boxplot = sns.boxplot(y=df[y_var], x=df[x_var], data=dataframe, whis=np.inf)
	sns_boxplot.set_xticklabels(sns_plot.get_xticklabels(),rotation = 90)
	sns_boxplot.tick_params(axis='x', labelsize=5)
	fig2 = sns_boxplot.get_figure()
	fig2.savefig('plots/' + y_var + '_' + x_var + '_boxplot.png', dpi = 300)
	



strings = ['period', 't', 'i', 'e', 'five', 'Shift.r', 'o', 'a', 'n', 'l']   # array to be used with scatterplot function below

# Creates 2D scatterplots of PPD vs RPD where PPD = H + UD  RPD = UD  
def scatter_plot(y_var, x_var, dataframe):
	y_label = y_var
	x_label = x_var + " + " + y_var
	x_label_save = x_var + "+" + y_var
	sns_scatterplot = sns.scatterplot(y=df[y_var], x=( df[x_var] + df[y_var] ), hue = 'subject', data=df, s=2)
	sns_scatterplot.set_xlabel(x_label)
	sns_scatterplot.set_ylabel(y_label)
	handles, labels = sns_scatterplot.get_legend_handles_labels()
	sns_scatterplot.legend(handles[:dataframe.subject.nunique()], labels[:dataframe.subject.nunique()])
	sns_scatterplot.legend(loc=4, prop={'size': 1})  #NEED TO MAKE LEGEND SMALLER!!!!
	sns_scatterplot.legend()
	fig3 = sns_scatterplot.get_figure()
	fig3.savefig('plots/' + y_label + '_' + x_label_save + '_scatterplot.png', dpi = 300)


# Creates a ROC curve for each user
def ROC_plot (subject, user_scores, imposter_scores):
	labels = [0]*len(user_scores) + [1]*len(imposter_scores)
	fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
	roc_auc = metrics.auc(fpr, tpr)
	plt.figure() 
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig('plots/ROC_' + str(subject) + '.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()


# Creates a FAR and FRR for each user
def EER_plot (subject, user_scores, imposter_scores):
	labels = [0]*len(user_scores) + [1]*len(imposter_scores)
	fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
	roc_auc = metrics.auc(fpr, tpr)
	missrates = 1 - tpr
	farates = fpr
	plt.figure()
	plt.title('EER Curve')
	plt.plot(missrates, '-')
	plt.plot(farates, '-')
	plt.legend(['missrates = FRR', 'farrates = FAR'], loc = 'upper left')
	plt.savefig('plots/EER_' + str(subject) + '.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()


# Plots the loss per epoch for each user
def loss_plot(x , y, subject):
	plt.plot(x) 
	plt.plot(y)
	plt.title('model')
	plt.xlabel('epoch')
	plt.legend(['loss', 'accuracy'], loc='upper left')
	#plt.show()
	plt.savefig('plots/loss_' + str(subject) + '.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()

# Plots the EER per user
def EER_bar_plot(data):

	names = list(data.keys())
	values = list(data.values())
	plt.figure()
	plt.bar(range(len(data)), values, tick_label=names)
	plt.title('EER per user')
	plt.xlabel('user', fontsize = 1)
	plt.xticks(rotation=90)
	plt.ylabel('EER')
	plt.draw()
	plt.savefig('plots/EER_per_user.png')
	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()





