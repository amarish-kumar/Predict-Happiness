#Import Required Packages
import os
import re
import numpy as np
import pandas as pd
import sklearn
import nltk

#Download Punkt and Stopwords Corpus
nltk.download('stopwords')
nltk.download('punkt')

#Import the downloaded Packages
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Import TF-IDF from scikit learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Store Stopwords in the stopword list
stopword = stopwords.words('english')

#Load Train data 
train_data = pd.read_csv('train.csv',sep=',')

#Preview it.
train_data.head(5)

#Length of Training Data
length = len(train_data)
print(length)

train_corpus = []
train_labels = np.zeros(length)

#Convert Categorial Labels into Numerical Labels
for i in range(length):
    train_corpus.append(train_data["Description"][i])
    if(train_data["Is_Response"][i]=="not happy"):
        train_labels[i] = 0
    else:
        train_labels[i] = 1


#Preprocess the data
train = []

def clean_data(sentence):
    sentence = re.sub(r'[^\w\s]','',sentence)
    sentence = re.sub(r'[^a-zA-Z\s]+','',sentence)
    token_data = word_tokenize(sentence)
    new_sent = ""
    #print(token_data)
    for i in range(len(token_data)):
        if token_data[i].lower() not in stopword:
            new_sent=new_sent+token_data[i].lower()+" "
    train.append(new_sent)

for i in range(len(train_corpus)):
    clean_data(train_corpus[i])

#Extract features using TF-IDF.
vectorizer  = TfidfVectorizer(min_df=1,encoding='utf-8', max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')

#Convert Train data to feature Vectors
train_corpus_tf_idf = vectorizer.fit_transform(train)

#Fit data using Logistic regression and SVM
from sklearn.linear_model import LogisticRegression
from sklearn import svm

lr = LogisticRegression()
lr.fit(train_corpus_tf_idf,train_labels)
lr.score(train_corpus_tf_idf,train_labels)

svmlinear = svm.SVC(kernel='linear')
svmlinear.fit(train_corpus_tf_idf,train_labels)

#Predict using SVM 
linearpred = svmlinear.predict(train_corpus_tf_idf)

#Check Accuracy
print(sum(linearpred==train_labels)/length)

#Load Test Data
test_data = pd.read_csv('test.csv',sep=',')
test_data.head(3)

length_test = len(test_data)
test_labels = np.zeros(length_test)
print(length_test)

#Copy Description to test array
test_corpus = []
for i in range(length_test):
    test_corpus.append(test_data["Description"][i])

#Clean Test data
test = []

def clean_data(sentence):
    sentence = re.sub(r'[^\w\s]','',sentence)
    sentence = re.sub(r'[^a-zA-Z\s]+','',sentence)
    token_data = word_tokenize(sentence)
    new_sent = ""
    #print(token_data)
    for i in range(len(token_data)):
        if token_data[i] not in stopword:
            new_sent=new_sent+token_data[i]+" "
    test.append(new_sent)

for i in range(length_test):
    clean_data(test_corpus[i])

#Transform the test data into feature vectors
test_corpus_tf_idf = vectorizer.transform(test)

#Make Predictions on Test data
test_predict = svmlinear.predict(test_corpus_tf_idf)

#Store Predictions in Submission File
import csv

#predictions using LR
with open('submission.csv', 'w') as csvfile:
    fieldnames = ['User_ID', 'Is_Response']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    count = 80132
    writer.writeheader()
    for predict in range(length_test):
        if(count==100000):
            num = 'id1e+05'
        else:
            num = 'id'+str(count)
        if(test_predict[predict]==1):
            label="happy"
        else:
            label="not_happy"
        writer.writerow({'User_ID': num, 'Is_Response': label})
        count+=1

#predictions using SVM
with open('submission_linear.csv', 'w') as csvfile:
    fieldnames = ['User_ID', 'Is_Response']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    count = 80132
    writer.writeheader()
    for predict in range(length_test):
        if(count==100000):
            num = 'id1e+05'
        else:
            num = 'id'+str(count)
        if(test_predict[predict]==1):
            label="happy"
        else:
            label="not_happy"
        writer.writerow({'User_ID': num, 'Is_Response': label})
        count+=1