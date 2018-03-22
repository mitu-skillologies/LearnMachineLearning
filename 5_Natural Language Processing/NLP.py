# Natural Language Processing

# Importing Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
# We can have Double Quotes, Commas but Tab is best! also we are ignoring double quotes

# Cleaning the Texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Creating a Corpus
corpus = []
for i in range(0, dataset.__len__()):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    nl = WordNetLemmatizer()
    review = [ps.stem(nl.lemmatize(word, pos='v')) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag Of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)

# Sparse Matrix -> CV
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting Data into Training & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

# Fitting Logistic Regression to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
