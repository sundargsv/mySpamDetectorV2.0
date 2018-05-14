# -*- coding: utf-8 -*-
"""
Created on Sat May 13 16:50:10 2018

@author: Sundar Gsv
"""
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


dataFrame = pd.read_csv('smsspamcorpus/SMSSpamCollection',sep='\t',names=['Status','Message'])

## dataFrame.head()
dataFrame.loc[dataFrame["Status"]=='ham',"Status",] = 1
dataFrame.loc[dataFrame["Status"]=='spam',"Status",] = 0

dataFrame_x = dataFrame["Message"]
dataFrame_y = dataFrame["Status"]

cv = TfidfVectorizer(min_df = 1,stop_words = 'english')

X_train, X_test, y_train, y_test = train_test_split(dataFrame_x, dataFrame_y, test_size = 0.2, random_state = 4)
## X_train.head()
X_trainCv = cv.fit_transform(X_train)
X_testCv = cv.transform(X_test)

classifier = MultinomialNB()
y_train = y_train.astype('int')
y_test = y_test.astype('int')

classifier.fit(X_trainCv, y_train)
predictions = classifier.predict(X_testCv)
print(accuracy_score(y_test, predictions))

## Predicting in real-time
##ToDO: Build some data preprocessing procedures
inputEmail = ["how are you"]
Xpred_cv = cv.transform(inputEmail)

result = classifier.predict(Xpred_cv)
print(result[0])
print(["Not Spam!", "Spam!"][not result[0]])

## testmessage=x_test.iloc[0]
## print - testmessage
## predictions=mnb.predict(x_testcv[0])
## predictions







