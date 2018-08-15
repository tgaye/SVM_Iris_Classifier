# SVM_Iris_Classifier
Uses variations of SVM's (Support Vector Machine) to classify the species of Iris in the popular Iris dataset, used in R.A. Fisher's 
classic 1936 paper: The Use of Multiple Measurements in Taxonomic Problems. I chose this data set due to its small sample size (150), 
widely noted as a condition SVM's deal well with.

### Import necessary Libaries
```
import pandas as pd
import numpy as np
import seaborn  as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
```

### Now we import our data, replace the directory link with wherever you choose to save the file on your computer. Check number of each Iris species.
```
df = pd.read_csv('../SVM_01/Iris.csv') # Read in data file of various Iris species
print("Number of Iris-setosa: {}".format(df[df.Species == 'Iris-setosa'].shape[0]))
print("Number of Iris-versicolor: {}".format(df[df.Species == 'Iris-versicolor'].shape[0]))
print("Number of Iris-virginica: {}".format(df[df.Species == 'Iris-virginica'].shape[0]))
print(df.shape)
```
Output:

Number of Iris-setosa: 50

Number of Iris-versicolor: 50

Number of Iris-virginica: 50


(150, 6)

We can see we have 150 entries, with 6 variables (including ID #).  We therefor have 5 variables to help with classification.

### Split data into Testing and Training
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
```
# Test 4 variations of SVM models (linear, rbm, polynomial, linearSVC)
```
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C), # Gamma optional hyperparam to opt. if rbf were best
          svm.SVC(kernel='poly', degree=3, C=C)) # Poly optional hyperparam to opt. if poly were best
```

Here is the output when we graph all four versions of our SVM (we find linear works the best, with an accuracy rate >99%).


