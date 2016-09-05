# Here for this project we will be analyzing Tweets about the
# Presidential Candidates by word, tokenizing them, and then using
# them to classify each overall tweet as either positive or negative
# Here we will import the Scikit Learn libraries among others necessary

import pandas as pd
import csv
import sys
import random
import numpy as np
import time
from time import strftime
#from pandas.DataFrame import query

import nltk
from nltk import corpus
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')
 
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

unlabeled_columns = [
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
    # Here we will parse a CSV file with the data on Row ID, Tweet ID,
    # Timestamp, President, Tweet

    training_file = csv.writer(open("training_data.csv", "wb+"))
    testing_file = csv.writer(open("testing_data.csv", "wb+"))
    unlabeled_file = csv.writer(open("unlabeled.csv", "wb+"))
    #tweet_file = csv.reader(open("tweets_two.csv", "rb"))
        
    # Since the Tensorflow DNN (Deep Neural Network) wants the
    # response variable in terms of numbers Here we will change the
    # labels into numbers (0 for positive, 1 for negative)
    
    labels.append("positive")
    labels.append("negative")
    labels.append("neutral")
    
    # This is how you randomize the data
    # Gotten from Github: 
    # (http://stackoverflow.com/questions/4618298/randomly-mix-lines-of-3-million-line-file)
    with open('tweets_two.csv','rb') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open('randomized_tweets.csv','wb+') as target:
        for _, line in data:
            target.write( line )
    
    tweet_file = csv.reader(open("randomized_tweets.csv", "rb"))
    index = 0
    
    # Now we will iterate through the randomized file and extract data
    # We need to get rid of the decimal points in the seconds columns
    # And then split up the data (2/3 train and 1/3 test)
    for row in tweet_file:
        (row_id, tweet_id, timestamp, president, tweet, label) = row
        raw_timestamp = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
        correct_format  = "%Y-%m-%d %H:%M:%S"
        timestamp = strftime(correct_format, raw_timestamp)
        ratio = 3
        
        # Take care of unlabeled data
        if label == "1":
            print(row_id)
            tokenize_row_write(unlabeled_file, row_id, tweet_id, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, "")
            continue
        
        
        write_to_csv = None
        if index % ratio == 0:
            write_to_csv = testing_file
        else:
            write_to_csv = training_file
            
        print(label)
        tokenize_row_write(write_to_csv, row_id, tweet_id, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, label)
            
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
    """ Building a Pipeline; this does all of the work in extract_and_train() for you """ 
    
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
 
    text_clf = text_clf.fit(x_train, y_train)
    print("Model trained")
    
    """ Evaluate performance on test set """
    
    predicted = text_clf.predict(x_test)
    #print("The accuracy of a Naive Bayes algorithm is: %d" % np.mean(predicted == y_test))
    #print("The accuracy of a Naive Bayes algorithm is: %d" % (1 - float(((y_test != predicted).sum()) / x_test.shape[0])))
    print("Number of mislabeled points out of a total %d points : %d"
          % (x_test.shape[0],(y_test != predicted).sum()))
    
    # Tune parameters
    parameter_tuning(text_clf, x_train, y_train)
    # Predict unlabeled tweets
    predict_unlabeled_tweets(text_clf)

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

def predict_unlabeled_tweets(classifier):
    predicted_tweets = csv.writer(open("predicted.csv", "wb+"))
    unlabeled_tweets = pd.read_csv("unlabeled.csv", names = unlabeled_columns)
    
    # Make predictions
    unlabeled_words = np.array(unlabeled_tweets["tweet"])
    predictions = classifier.predict(unlabeled_words)
    print(predictions)
    
    # Iterate through csv and get president and tweet
    # Add prediction to end
    for row, prediction in zip(unlabeled_tweets, predictions):
        #(row_id, tweet_id, day, hour, president, tweet, label) = row
        #predicted_tweets.writerow([president] + [tweet] + [prediction])
        predicted_tweets.writerow([row] + [prediction])



if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'parse':
            parse_csv()
        elif sys.argv[1] == 'train':
            extract_and_train()
        #elif sys.argv[1] == 'predict':
        #    predict_tweets()







