# -*- coding: utf-8 -*-
"""
ISTE:612.01: KPT Project (Event Track Mining Real Time Tweets for Predicting US Tornadoes)

System Integration of UI with Core Algorithm : This program Pre-Process the extracted tweets(Training dataset)
and perform supervised learning(using K-neighbor and NB classifier) based on input labelled dataset.
Comparison between performance of different classifier is shown in the program. Clutering is performed to get
visualize view of training dataset.

UI Interface : Made conenction with Flask Web development Framework .
Our System takes the Input Location from User and gives the exact prediction of Tornado occurance in that location.

@author : Utsav Dixit , Mayuresh Naik, Avni Taylor
"""

# -*- coding: utf-8 -*-
import pandas as pd
import got,codecs
from pymongo import MongoClient
import csv
import time
import random
from flask import render_template
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import nltk
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
import string
import numpy as np
from numpy._distributor_init import NUMPY_MKL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from flask import Flask, redirect, url_for, request

def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))

#Below line makes object of class Flask
app = Flask(__name__)

#Below block performs connection to MongoDB
client = MongoClient('localhost', 27017)
db = client['twitter_db']
print str(randomword(10))
collection = db[str(randomword(10))]


def preprocess_text(train):
    train_list = train['text'].values.tolist()
    train_list = train['hashtags'].values.tolist()
    lb = LabelEncoder()
    train['text'] = lb.fit_transform(train_list)
    train['hashtags'] = lb.fit_transform(train_list)
    return(train)

#Below function performs NB classification on tarining data and returns the output of prediction performed on Test data with accuracy
def NB(train_text_corpus, test_corpus):
    vectorizer = CountVectorizer(stop_words='english')
    train_text = vectorizer.fit_transform([tr[7] for tr in train_text_corpus])
    nb = MultinomialNB()
    nb.fit(train_text, [int(tr[2]) for tr in train_text_corpus])
    test_text = vectorizer.transform([ts[8] for ts in test_corpus])
    text_prediction = nb.predict(test_text)
    print(text_prediction)
    count = 0.0
    for i in range(len(text_prediction)):

        if text_prediction[i] == 1:
            count = count + 1
    print("\nNaive Bayes Classifier Results:")
    print("--------------------------------------------")
    print count
    print("Total Number of Test Documents=", len(text_prediction))
    print (count / len(text_prediction))*100
    return round((count / len(text_prediction))*100,2)

#Below module performs pre-processing on input stream of tweets.
def process_tweet_text(tweet,stopword,english_vocab):
   if tweet.startswith('@null'):
       return "[Tweet not available]"
   tweet = re.sub(r'\$\w*','',tweet) # Remove tickers
   tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet) # Remove hyperlinks
   tweet = re.sub(r'['+string.punctuation+']+', ' ',tweet) # Remove puncutations like 's
   twtok = TweetTokenizer(strip_handles=True, reduce_len=True)
   tokens = twtok.tokenize(tweet)
   tokens = [i.lower() for i in tokens if i not in stopword and len(i) > 2 and
                                             i in english_vocab]
   return tokens

#Below block extracts the tweets as per designated location entered by User via UI
def extract(name):
    print ("check point 1")
    since= str(datetime.date.today() - datetime.timedelta(3))
    until = str(datetime.date.today())
    print since
    print until
    tweetCriteria = got.manager.TweetCriteria().setNear(name).setWithin('100mi').setSince(since).setUntil(until).setQuerySearch('rain')
    tweetCriteria1 = got.manager.TweetCriteria().setNear(name).setWithin('100mi').setSince(since).setUntil(until).setQuerySearch('thunderstorm')
    tweetCriteria2 = got.manager.TweetCriteria().setNear(name).setWithin('100mi').setSince(since).setUntil(until).setQuerySearch('lightning')
    print ("check point 2")
    got.manager.TweetManager.getTweets(tweetCriteria, streamTweets)
    got.manager.TweetManager.getTweets(tweetCriteria1, streamTweets)
    got.manager.TweetManager.getTweets(tweetCriteria2, streamTweets)
    print ("check point 3")
    db= pd.DataFrame(list(collection.find()))
    print ("check point 4")
    if db.empty:
        return 0
    db.to_csv("Sample_1.csv",encoding='utf-8')
    print(db.head)
    print ("Data Loaded Successfully")

    with open('testingDataframe.csv', 'r')  as f:
        train_text_corpus = list(csv.reader(f))

    with open('Sample_1.csv', 'r')  as f2:
        test_corpus = list(csv.reader(f2))

    return NB(train_text_corpus, test_corpus)


#Below Block streams the tweet and extract the requested tweets based on location,username and ID
def streamTweets(tweets):
   for t in tweets:
      obj = {"user": t.username, "retweets": t.retweets, "favorites":
            t.favorites, "text":t.text,"geo": t.geo,"mentions":
            t.mentions, "hashtags": t.hashtags, "id":t.id
            }
      tweetind = collection.insert_one(obj)



@app.route('/success/<name>')
def success(name):
   result=extract(name)
   if(result==0):
       return render_template('results.html', name1 = name, result1=result)
   else:
       return render_template('results.html',name1=name, name = result)

@app.route('/home',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['search']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('search')
      return redirect(url_for('success',name = user))

#Below lines used to run Flask Environment
if __name__ == '__main__':
   app.run(debug = True)
