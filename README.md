# SVM_Iris_Classifier
Uses variations of SVM's (Support Vector Machine) to classify the species of Iris in the popular Iris dataset, used in R.A. Fisher's 
classic 1936 paper: The Use of Multiple Measurements in Taxonomic Problems. I chose this data set due to its small sample size (150), 
widely noted as a condition SVM's deal well with.

Import necessary Libaries
'''
import pandas as pd
import numpy as np
import seaborn  as sns
import matplotlib.pyplot as plt
'''
#Now we import our data, replace the directory link with wherever you choose to save the file on your computer.
'''
df = pd.read_csv('../SVM_01/Iris.csv') # Read in data file of various Iris species
print("Number of Iris-setosa: {}".format(df[df.Species == 'Iris-setosa'].shape[0]))
print("Number of Iris-versicolor: {}".format(df[df.Species == 'Iris-versicolor'].shape[0]))
print("Number of Iris-virginica: {}".format(df[df.Species == 'Iris-virginica'].shape[0]))
'''
