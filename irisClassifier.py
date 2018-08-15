# This code uses variations of SVM's (Support Vector Machine) to classify the species of Iris in the popular
# Iris dataset, used in R.A. Fisher's classic 1936 paper: The Use of Multiple Measurements in Taxonomic Problems.
# I chose this data set due to its small sample size (150), which is widely noted as a condition SVM's deal well with.

# Import needed libraries
import pandas as pd
import numpy as np
import seaborn  as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../SVM_01/Iris.csv') # Read in data file of various Iris species
corr = df.corr() # Create Matrix of Correlations
print(df.isnull().sum()) # Check for Null Values
print(df.shape) # Check shape of data
# We have 150 flowers x 6 variables (including ID numbers)

# Check population size of each species (3)
print("Number of Iris-setosa: {}".format(df[df.Species == 'Iris-setosa'].shape[0]))
print("Number of Iris-versicolor: {}".format(df[df.Species == 'Iris-versicolor'].shape[0]))
print("Number of Iris-virginica: {}".format(df[df.Species == 'Iris-virginica'].shape[0]))
# 150 total flowers, 50 of each species.

# Seperating Features and Labels (i.e removing labels)
X=df.iloc[:, :-1]

#Transform label(species) strings into Ints
from sklearn.preprocessing import LabelEncoder
y=df.iloc[:,-1]

# Encode label category
# Iris-setosa -> 0
# Iris-versicolor -> 1
# Iris-virginica -> 2
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)

# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split data into Testing and Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Run SVM with default hyperparameters
from sklearn.svm import SVC
from sklearn import metrics

# Check accuracy of 3 variations of kernel transformations using k-fold validation.
# K-fold method used as oppose to splitting data into 2 sets due to the small sample size to begin with.
# CV = number of items we subsample, cant be greater than number of items in each class (50 in this case).
# The higher CV of subsample size the more confident we can be.
from sklearn.model_selection import cross_val_score
svc=SVC(kernel='linear',C=0.1) # C hyperparam tells the SVM how much you want to avoid misclassifying each training example.
scores = cross_val_score(svc, X, y, cv=35, scoring='accuracy')
print('Mean accuracy of Linear kernel: ', scores.mean())

svc=SVC(kernel='rbf',C=0.1) # C hyperparam tells the SVM how much you want to avoid misclassifying each training example.
scores = cross_val_score(svc, X, y, cv=35, scoring='accuracy')
print('Mean accuracy of RBF kernel: ', scores.mean())

svc=SVC(kernel='poly',C=0.1) # C hyperparam tells the SVM how much you want to avoid misclassifying each training example.
scores = cross_val_score(svc, X, y, cv=35, scoring='accuracy')
print('Mean accuracy of Polynomial kernel: ', scores.mean())

# We can loop through to optimize C for our best kernel function.
C_range=list(np.arange(0.1,1,0.1))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=35, scoring='accuracy')
    acc_score.append(scores.mean())
print('Accuracy for each C value (.1 -> 1:) ', acc_score) # It appears any number greater than C=.1 works best.

# Check linear again, with optimal C this time.
svc=SVC(kernel='linear',C=0.2) # C hyperparam tells the SVM how much you want to avoid misclassifying each training example.
scores = cross_val_score(svc, X, y, cv=35, scoring='accuracy')
print('Mean of optimal Linear kernel: ', scores.mean())
# It appears we do gain a marginal boost in accuracy with the optimal C.  Model converges on 1 more often than not.

#---------------------------------------------------------------
# Helper functions for k-fold cross validation
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
#---------------------------------------------------------------

# Now lets visualize our data/classifier.
from sklearn import svm, datasets
iris = datasets.load_iris()

# Take the first two features. We could avoid this by using a two-dim dataset or performing dim. reduction (PCA)
X2 = iris.data[:, :2]
y2 = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 0.2  # SVM regularization parameter, using .2 because we found value >.1 to work best.
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C), # Gamma optional hyperparam to opt. if rbf were best
          svm.SVC(kernel='poly', degree=3, C=C)) # Poly optional hyperparam to opt. if poly were best
models = (clf.fit(X2, y2) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X2[:, 0], X2[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y2, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max()) # Use min and max to set limits
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

