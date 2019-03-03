# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 08:05:10 2019

@author: dhane
"""

import pandas as pd
#Function to create DTM from BoW model
from sklearn.feature_extraction.text import CountVectorizer
#Tokenizer that removes unwanted elements
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
#Module for accuracy calculation
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
#from sklearn.feature_extraction.text import TfidfVectorizer

data=pd.read_csv('uci-news-aggregator.csv')
data.head()
#print(data['CATEGORY']=='b')
data.CATEGORY.value_counts()

token=RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts=cv.fit_transform(data['TITLE'])
#print(text_counts)

X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['CATEGORY'], test_size=0.01, random_state=1)

clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(predicted, y_test))