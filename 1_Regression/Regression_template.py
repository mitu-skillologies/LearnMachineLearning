# REGRESSION TEMPLATE

# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting Data into Training & Testing
'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("POST-SCALED VALUES")
print(X_train)
print(X_test)
# DO YOU NEED TO SCALE DUMMY VARIABLES?
# IT DEPENDS on your models
# DO YOU NEED TO SCALE Y Variables?
# NO, Cz IT is dependent variable
"""

# Fitting the Regression  Model to the Dataset
# Create your new Regressor here!

# Predicting a new Result
y_pred = regressor.predict(6.5)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(), color='black')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# For Higher Resolution nd smoother curve
# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(), color='black')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
