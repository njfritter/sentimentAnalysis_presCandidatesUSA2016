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

import nltk
"""
from nltk import corpus
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
"""
def extract_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words
            
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

file = "tweets.csv"

columns = [
    "row_id",
    "tweet_id",
    "timestamp",
    "president",
    "tweet"
    ]

train_columns = [
    "row_id",
    "tweet_id",
    "day",
    "month",
    "president",
    "tweet",
    "label"
    ]

test_columns = [
    "row_id",
    "tweet_id",
    "day",
    "month",
    "president",
    "tweet",
    "label"
    ]

categorical_columns = [
    "row_id",
    "tweet_id",
    "day",
    "month",
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
test_tweets = [
    (['feel', 'happy', 'this', 'morning'], 'positive'),
    (['larry', 'friend'], 'positive'),
    (['not', 'like', 'that', 'man'], 'negative'),
    (['house', 'not', 'great'], 'negative'),
    (['your', 'song', 'annoying'], 'negative')]
"""

word_features = get_word_features(get_words_in_tweets(tweets))
print("word features made")
training_set = nltk.classify.apply_features(extract_features(tweets, word_features))
print("Training set made")
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classified")
tweet = 'Larry is my friend'
print extract_features(tweet.split())
def train(labeled_featuresets, estimator=ELEProbDist):
    # Create the P(label) distribution
    label_probdist = estimator(label_freqdist)
    
    # Create the P(fval|label, fname) distribution
    feature_probdist = {}
    return NaiveBayesClassifier(label_probdist, feature_probdist)
print feature_probdist

print feature_probdist[('negative', 'contains(best)')].prob(True)

print classifier.show_most_informative_features(32)



def parse_csv():
    # Here we will parse a json file with trello data inside into a CSV file
    # With the data on Card Id, Title, Descriptions, and Labels
    # with open(file) as f:
    #    data = json.load(f)

    train_file = csv.writer(open("train.csv", "wb+"))
    test_file = csv.writer(open("test.csv", "wb+"))
    # unlabeled_file = csv.writer(open("unlabeled.csv", "wb+"))
    tweet_file = csv.reader(open("tweets.csv", "rb"))

    # Write headers
    train_file.writerow(train_columns)
    test_file.writerow(test_columns)


    # tweets = tweet_file["json_output"]
    # Since the Tensorflow DNN (Deep Neural Network) wants the
    # response variable in terms of numbers Here we will change the
    # labels into numbers (0 for positive, 1 for negative)

    labels.append(0, 1)

    # Now shuffle them
    random.shuffle(tweet_file)

    index = 0
    
    for row in tweet_file:
        row = (row_id, tweet_id, timestamp, president, tweet)
        timestamp = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        # Here we will split up the data into 2/3 training and 1/3 test
        ratio = 3

        """
        if len(card['labels']) == 0:
            print(card['id'])
            #unlabeled_file.writerow([card['id'], tokenize_row_write(unlabeled_file, card['name'], card['desc'], "")])
            #unlabeled_file.writerow([card['id'], card['name'], ""])
            tokenize_row_write(unlabeled_file, card['id'], card['name'], card['desc'], "")
            continue
            """
        
        write_to_file = None
        if index % ratio == 0:
            write_to_file = test_file
        else:
            write_to_file = train_file

        label = row[1]
        print(label)
        tokenize_row_write(write_to_file, row_id, tweet_id, timestamp.tm_wday, timestamp.tm_hour, president, tweet)

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
def tokenize_row_write(file_csv_writer, tweet, card_name, card_desc, label):
    words_name = tokenized_string(card_name)
    words_desc = tokenized_string(card_desc)

    words = words_name + words_desc

    for word in words:
        file_csv_writer.writerow([card_id, word, label])

    #file_csv_writer.writerow([words] + [label])



