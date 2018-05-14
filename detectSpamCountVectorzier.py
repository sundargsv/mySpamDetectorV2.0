# -*- coding: utf-8 -*-
"""
Created on Sat May 13 12:31:19 2018

@author: Sundar Gsv
"""
import pandas as pd
from  sklearn.feature_extraction.text  import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

dataFrame = pd.read_csv('smsspamcorpus/SMSSpamCollection',sep='\t',names=['Status','Message'])

## dataFrame.head()
dataFrame.loc[dataFrame["Status"]=='ham',"Status",] = 1
dataFrame.loc[dataFrame["Status"]=='spam',"Status",] = 0

dataFrame_x = dataFrame["Message"]
dataFrame_y = dataFrame["Status"]

cv = CountVectorizer()

X_train, X_test, y_train, y_test = train_test_split(dataFrame_x, dataFrame_y, test_size = 0.2, random_state = 4)

## X_train.head()
""" x_traincvTest = cv.fit_transform(["Hi How are you How are you doing","Hi what's up","Wow that's awesome"])
x_traincvTest.toarray()
cv.get_feature_names() """

## Transforming the BOW/ docs to numbers while y_train is already one (0's and 1's) 
## use fit_transform only when handling with training data not with test data
X_trainCv = cv.fit_transform(X_train)
arrX_train = X_trainCv.toarray()

## Example to test and learn
"""arrX_train[0]
cv.inverse_transform(arrX_train[0])
X_train.iloc[0] """

## Transforming the BOW/ docs to numbers while y_test is already one (0's and 1's)
X_testCv = cv.transform(X_test)
arrX_test = X_testCv.toarray()

## Example to test and learn
"""arrX_test[0]
cv.inverse_transform(arrX_test[0])
X_test.iloc[0] """

classifier = MultinomialNB()

y_train = y_train.astype('int')
y_test = y_test.astype('int64')
## y_train

classifier.fit(X_trainCv, y_train)

## Test
predictions = classifier.predict(X_testCv)

## Do run the below command to get rid of error at the time of finding accuracy_score
## ser = pd.Series([1, 2], dtype='int64')
print(accuracy_score(y_test, predictions))

## Predicting in real-time
##ToDO: Build some data preprocessing procedures
inputEmail = ["please click here www.link.com to avail the offer"]
Xpred_cv = cv.transform(inputEmail)

result = classifier.predict(Xpred_cv)
print(result[0])
print(["Not Spam!", "Spam!"][not result[0]])










