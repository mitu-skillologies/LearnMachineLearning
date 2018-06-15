# Data Pre-Processing
# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

print("FIRST X ___\n", X)
print("FIRST y ___\n", y)

# Taking care of Missing Data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer1 = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print("IMPUTED X ____\n",X)

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Learn the Output
# PROBLEM for Country & Purchase
# France > Spain > Germany! Well... WTH is that!

print("TRANSFORMED X____\n",X)

# Solution
# COUNTRY
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# PURCHASE
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print("CATEGORICAL y",y)

# Splitting Data into Training & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

print("PRE-SCALED VALUES")
print(X_train)
print(X_test)
print("_"*40)
print(y_train)
print(y_test)

# Feature Scaling
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
# NO, Cz IT is a dependent variable