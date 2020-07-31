import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the Data
df = pd.read_csv("news.csv")

# Get shape and head
shape = df.shape
print(df.head())

# Get The Labels
labels = df.label

# Split into test and train
x_train, x_test, y_train, y_test = train_test_split(
    df['text'],
    labels,
    test_size=0.2,
    random_state=7
)

# Initialize a TfidfVctorizer
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)
# Stop words are the most common words in a language that are to be filtered out
# df : document frequency --> terms with df more than specified value will be discarded

# Fit and Transform Train Set, Transform Test Set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate the accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Build Confucion Matrix
cf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print(cf_matrix)



