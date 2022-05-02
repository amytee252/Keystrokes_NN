# Keystroke Dynamics Analysis and Prediction

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

#TABS NOT SPACES!!!!

df = pd.read_csv("./dataset/DSL-StrongPasswordData.csv") #open csv file for reading in pandas dataframe

print(df.info()) #Get info about dataframe
print(df) #look at first handful of entries to know what we are dealing with

df_original = df.copy() #Always keep an original copy of the dataframe before any changes are made to it, just in case it is useful in the future!

#Lets plot a few things to understand what the data looks like
### plot some things!


def swarmplot(y_var, x_var, dataframe):
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

#DO NOT UNCOMMENT THE BELOW! IT HAS ALREADY BEEN RUN AND UNCOMMENTING THIS WILL WASTE HOURS OF YOUR LIFE!!!!
'''
for column in df.columns[3:]:
	swarmplot(str(column), 'subject', df)
'''	
#Scatterplot of PPD vs RPD  PPD = H + UD  RPD = UD  password = .tie5Roanl
def scatterplot(y_var, x_var, dataframe):
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

strings = ['period', 't', 'i', 'e', 'five', 'Shift.r', 'o', 'a', 'n', 'l']

for i in range(len(strings) - 1):
	string_first = strings[i]
	string_second = strings[i+1]
	scatterplot('UD.' + string_first + '.' + string_second, 'H.' + string_second, df)


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
	df_train_dict[subject] = df_temp_dict[subject].sample(n=200)  # Instead of taking first 200 I am sampling randomly
	df_test_dict[subject] = df_temp_dict[subject].drop(df_train_dict[subject].index) 
	imposter_data = df.loc[df.subject != subject, :]  
	df_imposter_dict[subject] = imposter_data.groupby("subject").head(5)#.loc[:, "H.period":"H.Return"]
	if subject is  50:
		print("Last user's dataframe")
		print(df_imposter_dict[subject])

	df_train_dict[subject] = pd.DataFrame(scaler.fit_transform(df_train_dict[subject]), columns=df_temp_dict[subject].columns)
	df_test_dict[subject] = pd.DataFrame(scaler.fit_transform(df_test_dict[subject]), columns=df_temp_dict[subject].columns)
	df_imposter_dict[subject] = pd.DataFrame(scaler.fit_transform(df_imposter_dict[subject]), columns=df_temp_dict[subject].columns)

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


model = tf.keras.Sequential([  #build NN
			tf.keras.layers.Dense(31, activation='relu'),   #Change it to get 31 from length of columns
			tf.keras.layers.Dense(8 , activation='relu'),  #Maybe add in a dropout layer? 20% dropout to prevent against overfitting
			tf.keras.layers.Dense(1 , activation='sigmoid')
])

'''

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x, y, epochs=100, batch_size=64)


plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#Loop over subjects
#convert each dataframe to a tensorflow tensor
#train
#test with test
#test with imposters


#Train is train

'''
#test is test and again test with imposter
