# POLYNOMIAL LINEAR REGRESSION

# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv("1_Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting Data into Training & Testing
# WE WILL NOT SPLIT DATA HERE Since the size of dataset is very small
# and that'll be ridiculous to do that with distinct values

# Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
d = lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='black')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='black')
plt.title("<?> Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new Result with Linear Regression
o = lin_reg.predict(X =6.5)

# Predictig a new Result with Polynomial Regression
o1 = lin_reg_2.predict(poly_reg.fit_transform(6.5))
