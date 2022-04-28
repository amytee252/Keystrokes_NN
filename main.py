# Keystroke Dynamics Analysis and Prediction

import numpy as np
import pandas as pd

import matplotlib as plt

import seaborn as sns


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

