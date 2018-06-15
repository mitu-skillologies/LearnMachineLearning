'''
BUSINESS PROBLEM DESCRIPTION:

We are Resolving a problem of Bank Churning, meaning we have to find the reason why Customers are
Leaving bank. Clearly we can't bring back but we need to find a pattern and target those customers
who will be most likely be leaving the Bank
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('DataKit/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

# -----------------------------------------------------------------

# Import KERAS Library
import keras
from keras.models import Sequential # Initialize ANN
from keras.layers import Dense # Build Layers of ANN

# Initialize the ANN
classifier = Sequential()

# Adding the Input Layer and the First Hidden Layer
# Rectified Linear Unit - Relu
classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation= 'relu', input_dim= 11))

# STEP 6
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation= 'relu'))

# Adding the Output Layer
# Sigmoid - Sigmoid Function
classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation= 'sigmoid'))
# If your DV is oneHotEncoded,
# then you will need to have 3 / 4 / 5 units and activation function would be softmax

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# optimizer = Algorithm
# loss function = IF DV has bin op binary_crossentropy IF more then categorical_crossentropy
# After each batch ACCURACY increases the Batch processing performance

# EPOCH = Whole Training set passed thru ANN on Dataset
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# -----------------------------------------------------------------

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()