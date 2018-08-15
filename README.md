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
### Output:

Number of Iris-setosa: 50

Number of Iris-versicolor: 50

Number of Iris-virginica: 50


(150, 6)

We can see we have 150 entries, with 6 variables (including ID #).  We therefor have 5 variables to help with classification.
Because we have 5 variables (multivariable) and are attempting linear regression with our SVM, we must use what is known as a "kernel trick", applying a kernel function to our model to allow for linear analysis on multidimensional data.


### Check which kernel function works best for this data set (Linear, RBF, Polynomial)
```
svc=SVC(kernel='linear',C=0.1) # C hyperparam tells the SVM how much you want to avoid misclassifying each training example.
scores = cross_val_score(svc, X, y, cv=35, scoring='accuracy')
print('Mean accuracy of Linear kernel: ', scores.mean())

svc=SVC(kernel='rbf',C=0.1) 
scores = cross_val_score(svc, X, y, cv=35, scoring='accuracy')
print('Mean accuracy of RBF kernel: ', scores.mean())

svc=SVC(kernel='poly',C=0.1) 
scores = cross_val_score(svc, X, y, cv=35, scoring='accuracy')
print('Mean accuracy of Polynomial kernel: ', scores.mean())
```
### Output:

Mean accuracy of Linear kernel:  0.9952380952380953

Mean accuracy of RBF kernel:  0.9904761904761906

Mean accuracy of Polynomial kernel:  0.9095238095238096

Liner kernel wins!


### We have a single hyperparameter that we can optimize, namely C. 
```
C_range=list(np.arange(0.1,1,0.1))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=35, scoring='accuracy')
    acc_score.append(scores.mean())
print('Accuracy for each C value (.1 -> 1:) ', acc_score) # It appears any number greater than C=.1 works best.
```
Now that we have optimal parameters, its time for some modeling:


### Split data into Testing and Training
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
```
### Test 4 variations of SVM models 
```
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C), # Gamma optional hyperparam to opt. if rbf were best
          svm.SVC(kernel='poly', degree=3, C=C)) # Poly optional hyperparam to opt. if poly were best
```

Here is the output when we graph all four versions of our SVM (we find linear works the best, with an accuracy rate >99%).
Boundries drawn using first 2 variables (Sepal Length, Sepal Width) of our 5 total variables.

![figure_1](https://user-images.githubusercontent.com/34739163/44144655-9e7784d6-a045-11e8-9713-6c9846f3f159.png)

We can see visual confirmation that the Linear kernel function works best of all the SVM models.  When we use cross-validation
with optimal C param of 0.2, we get perfect classification (1.0) of our small data set (150 samples).
It is clear SVM's can perform quite well with small sample sizes.
