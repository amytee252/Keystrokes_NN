# Keystroke Dynamics Analysis and Prediction

import numpy as np
import pandas as pd

import matplotlib as plt

import seaborn as sns


df = pd.read_csv("./dataset/DSL-StrongPasswordData.csv") #open csv file for reading in pandas dataframe

print(df.info()) #Get info about dataframe
print(df) #look at first handful of entries to know what we are dealing with
