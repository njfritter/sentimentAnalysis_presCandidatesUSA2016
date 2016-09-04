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
"""

stopset = list(set(stopwords.words('english')))

def word_feats(words):
    return dict([(word, True) for word, sentiment in words.iteritems() if word not in stopset])

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words
            
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

"""
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

"""
# A set of positive and negative tweets
pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]
neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

print("Tweets made")
    
# Now the tweets are all tokenized by word, with the sentiment
# (positive/negative) next to it
    
# And now a list with test tweets
test_tweets = dict.fromkeys(['feel', 'happy', 'this', 'morning'], 'positive')
test_tweets.update(dict.fromkeys(['larry', 'friend'], 'positive'))
test_tweets.update(dict.fromkeys(['not', 'like', 'that', 'man'], 'negative'))
test_tweets.update(dict.fromkeys(['house', 'not', 'great'], 'negative'))
test_tweets.update(dict.fromkeys(['your', 'song', 'annoying'], 'negative'))
    
# We use the format above because the NLTK classifier only accepts Python dictionaries 
# Lists or other "mutable" sets of data will not work
# Basically the keys must stay constant through the process  
# And dictionaries accomplish this

test_tweets_two = [
    (['feel', 'happy', 'this', 'morning'], 'positive'),
    (['larry', 'friend'], 'positive'),
    (['not', 'like', 'that', 'man'], 'negative'),
    (['house', 'not', 'great'], 'negative'),
    (['your', 'song', 'annoying'], 'negative')]


# Time to change the test_tweets into the proper format
test_feats = word_feats(test_tweets)

print(test_feats)

test_feats_two = [
    (dict('feel', 'happy', 'this', 'morning')),
    (dict('larry', 'friend')),
    (dict('not', 'like', 'that', 'man')),
    (dict('house', 'not', 'great')),
    (dict('your', 'song', 'annoying')),
    ]    
"""


def parse_csv():
    # Here we will parse a json file with trello data inside into a CSV file
    # With the data on Card Id, Title, Descriptions, and Labels
    # with open(file) as f:
    #    data = json.load(f)

    train_file = csv.writer(open("train.csv", "wb+"))
    test_file = csv.writer(open("test.csv", "wb+"))
    unlabeled_file = csv.writer(open("unlabeled.csv", "wb+"))
    tweet_file = csv.reader(open("tweets_two.csv", "rb"))

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
    X_train_counts = count_vect.fit_transform(train['tweet'])
    print(X_train_counts.shape)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)

    """ Training a classifer """

    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    print("Model fitted")

    test_words = ['Our plan promises to revive the economy and build jobs. Trump will destroy the middle class and feed his pockets']
    X_new_counts = count_vect.transform(test_words)
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
    print("The accuracy of a Naive Bayes algorithm is: %d" % np.mean(predicted == y_test))
    print("Number of mislabeled points out of a total %d points : %d"
          % (len(x_test),(y_test != predicted).sum()))

    parameter_tuning(text_clf)


parse_csv()
extract_and_train()
"""
word_features = get_word_features(get_words_in_tweets(tweets))
print("word features made")
#training_set = nltk.classify.apply_features(extract_features(tweets, word_features))
training_set = nltk.classify.apply_features(extract_features, tweets)
print("Training set made")
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classified")
tweet = 'Larry is my friend'
print extract_features(tweet.split())
print classifier.classify(extract_features(tweet.split()))

keys = []
for tweet in test_tweets:
    print(tweet)
    print(classifier.classify(extract_features(tweet.split())))
#classifier.classify_many(test_feats)

#for pdist in classifier.prob_classify_many(test_feats):
#    print('%.4f %.4f' % (pdist.prob('positive'), pdist.prob('negative')))
print 'accuracy: ', nltk.classify.util.accuracy(classifier, test_feats)
classifier.show_most_informative_features()


def train(labeled_featuresets, estimator=ELEProbDist):
    # Create the P(label) distribution
    label_probdist = estimator(label_freqdist)
    
    # Create the P(fval|label, fname) distribution
    feature_probdist = {}
    return NaiveBayesClassifier(label_probdist, feature_probdist)

print feature_probdist

print feature_probdist[('negative', 'contains(best)')].prob(True)

print classifier.show_most_informative_features(32)
"""




