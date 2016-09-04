# Here for this project we will be analyzing Tweets about the
# Presidential Candidates by word, tokenizing them, and then using
# them to classify each overall tweet as either positive or negative
# Here we will import the Tensorflow libraries among others necessary

import tempfile
import pandas as pd
import tensorflow as tf
import json
import csv
import sys
import random
import numpy as np
import time
from time import strftime

import nltk
from nltk.probability import FreqDist
from nltk import corpus
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
 #nltk.download('punkt')
 #nltk.download('stopwords')
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.corpus import CategorizedPlaintextCorpusReader
 
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV


file = "tweets_two.csv"

columns = [
    "row_id",
    "tweet_id",
    "timestamp",
    "president",
    "tweet",
    "label"
    ]

train_columns = [
    "row_id",
    "tweet_id",
    "day",
    "hour",
    "president",
    "tweet",
    "label"
    ]

test_columns = [
    "row_id",
    "tweet_id",
    "day",
    "hour",
    "president",
    "tweet",
    "label"
    ]

categorical_columns = [
    "row_id",
    "tweet_id",
    "day",
    "hour",
    "president",
    "tweet",
    "label"
    ]
continuous_columns = []

labels = []

local_stopwords = []

def parse_csv():
    # Here we will parse a json file with trello data inside into a CSV file
    # With the data on Card Id, Title, Descriptions, and Labels
    # with open(file) as f:
    #    data = json.load(f)
    
    train_file = csv.writer(open("train.csv", "wb+"))
    test_file = csv.writer(open("test.csv", "wb+"))
    unlabeled_file = csv.writer(open("unlabeled.csv", "wb+"))
    tweet_file = csv.reader(open("tweets_two.csv", "rb"))
    print("Files read")
    """ 
    # Write headers
    train_file.writerow(train_columns)
    test_file.writerow(test_columns)
    """
    
    # Since the Tensorflow DNN (Deep Neural Network) wants the
    # response variable in terms of numbers Here we will change the
    # labels into numbers (0 for positive, 1 for negative)
    
    labels.append("positive")
    labels.append("negative")
    labels.append("neutral")
    
    # Now shuffle them
    # random.shuffle(tweet_file)
    
    index = 0
    
    for row in tweet_file:
        (row_id, tweet_id, timestamp, president, tweet, label) = row
        raw_timestamp = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
        correct_format  = "%Y-%m-%d %H:%M:%S"
        timestamp = strftime(correct_format, raw_timestamp)
         # Here we will split up the data into 2/3 training and 1/3 test
        ratio = 3
        
        if len(label) == 0:
            print(row_id)
             #unlabeled_file.writerow([card['id'], tokenize_row_write(unlabeled_file, card['name'], card['desc'], "")])
             #unlabeled_file.writerow([card['id'], card['name'], ""])
            tokenize_row_write(unlabeled_file, row_id, tweet_id, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, "")
            continue
        
        
        write_to_file = None
        if index % ratio == 0:
            write_to_file = test_file
        else:
            write_to_file = train_file
            
        print(label)
        tokenize_row_write(write_to_file, row_id, tweet_id, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, label)
            
        index += 1

def clean_word(word):
    return word not in stopwords.words('english') and word not in local_stopwords

def tokenized_string(sent):
    tokenizer = RegexpTokenizer(r'\w+')
    words = [word.lower() for word in tokenizer.tokenize(sent)]
    words = [word for word in words if clean_word(word)]
    
    return words

# Tokenize the title and description, then write everything to the corresponding
# csv file
def tokenize_row_write(file_csv_writer, row_id, tweet_id, day, hour, president, tweet, label):
    words_tweet = tokenized_string(tweet)

    file_csv_writer.writerow([row_id] + [tweet_id] + [day] + [hour] + [president]+ [words_tweet] + [label])
    
    """
    for word in words:
    file_csv_writer.writerow([card_id, word, label])
    """    
    
 
def extract_and_train():
        
     #tweet_file = pd.read_csv(file, names = columns)
    
    train = pd.read_csv("train.csv", names = train_columns)
    test = pd.read_csv("test.csv", names = test_columns)
    
    x_train = np.array((train['row_id'], train['tweet_id'], train['day'], train['hour'], train['president'], train['tweet']))
    y_train = np.array(train['label'])
    
    x_test = np.array((test['row_id'], test['tweet_id'], test['day'], test['hour'], test['president'], test['tweet']))
    y_test = np.array(test['label'])
    
    train_words = np.array(train['tweet'])
    test_words = np.array(test['tweet'])
    print("Data read")
    
    """ Extract features from text files """
    
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_words)
    print(X_train_counts.shape)
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)
    
    """ Training a classifer """
    
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    print("Model fitted")
    
    test_sentence = ['Our plan promises to revive the economy and build jobs. Trump will destroy the middle class and feed his pockets']
    X_new_counts = count_vect.transform(test_sentence)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    print("Test data transformed")
    
    predicted = clf.predict(X_new_tfidf)
    print("Test data predicted")

    for words, category in zip(test_words, predicted):
        print('%r => %s' % (words, category))
        
    naive_bayes(train_words, y_train, test_words, y_test)
 
def naive_bayes(x_train, y_train, x_test, y_test):
    """ Building a Pipeline; this does all of the work in train_NB() for you """ 
    
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
 
    text_clf = text_clf.fit(x_train, y_train)
    print("Model trained")
    
    """ Evaluate performance on test set """
    
    predicted = text_clf.predict(x_test)
     #print("The accuracy of a Naive Bayes algorithm is: %d" % np.mean(predicted == y_test))
    print("The accuracy of a Naive Bayes algorithm is: %d" % float((((y_test != predicted).sum()) / x_test.shape[0])))
    print("Number of mislabeled points out of a total %d points : %d"
          % (x_test.shape[0],(y_test != predicted).sum()))
    
    parameter_tuning(text_clf, x_train, y_train)
    
def parameter_tuning(text_clf, x_train, y_train):
    """ Classifiers can have many different parameters that can make the                                                                                                                   
    algorithm more accurate (MultinomialNB() has a smoothing                                                                                                                               
    parameter, SGDClassifier has a penalty parameter, etc.). Here we                                                                                                                       
    will run an exhaustive list of the best possible parameter values """
    
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }
    
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    
    gs_clf = gs_clf.fit(x_train, y_train)
    
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
        
        
        print(score)
        
        """                                                                                                                                                                                    
        print(metrics.classification_report(y_test, predicted,                                                                                                                                 
        target_names=twenty_test.target_names))                                                                                                                                                
        """

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'parse':
            parse_csv()
        elif sys.argv[1] == 'train':
            extract_and_train()
        







