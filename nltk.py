# Here for this project we will be analyzing Tweets about the
# Presidential Candidates by word, tokenizing them, and then using
# them to classify each overall tweet as either positive or negative
# Here we will import the Tensorflow libraries among others necessary

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import pandas as pd
import tensorflow as tf
import json
import csv
import sys
import random
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')

file = "output.csv"

train_columns = [
    "word",
    "label"
]

test_columns = [
    "word",
    "label"
]

categorical_columns = [
    "word"
]
continuous_columns = []

labels = []

local_stopwords = []


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
    words_filtered = [word.lower() for word in words.split() if len(word) >= 3] 
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


def parse_csv():
    # Here we will parse a json file with trello data inside into a CSV file
    # With the data on Card Id, Title, Descriptions, and Labels
    # with open(file) as f:
        
    train_file = csv.writer(open("train.csv", "wb+"))
    test_file = csv.writer(open("test.csv", "wb+"))
    # unlabeled_file = csv.writer(open("unlabeled.csv", "wb+"))

    """
    Write headers
    train_file.writerow(train_columns)
    test_file.writerow(test_columns)
    """

    # Since the Tensorflow DNN (Deep Neural Network) wants the response variable in terms of numbers
    # Here we will change the labels into numbers (0 for positive, 1 for negative)

    labels.append(0, 1)

    # Now shuffle them
    random.shuffle(tweets)

    index = 0

    for row in tweets:
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
        tokenize_row_write(write_to_file, index, card['name'], card['desc'], label)

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


# Now let's combine all the tweets


def categorize(model_dir, model):
    """
    Here we will be more specific with features and fill them in
    In other words we will be selecting and engineering features for the model
    Some features will only have a few options (spare_columns_with_keys() or spare_column_with_hash_bucket))
    Some continuous features will be turned into categorical features through bucketization (when there is NOT a linear relationship between a continuous feature and a label)
    And we should also look into the differences between different features combinations (explanatory variables):
    """

    word_hashed = tf.contrib.layers.sparse_column_with_hash_bucket("word", hash_bucket_size = 1000)
    # label_hashed = tf.contrib.layers.sparse_column_with_keys(column_name="label", keys=labels)

    # Now we will make our sets of wide and deep columns
    wide_columns = [
        word_hashed,
        # label_hashed,
        # tf.contrib.layers.crossed_column([word_hashed, label_hashed], hash_bucket_size=int(1e4))
    ]
    deep_columns = [
        tf.contrib.layers.embedding_column(word_hashed, dimension = 1),
    ]

    # These are the number of classes we are trying to predict
    num_classes = 10

    # Here we will build a Logistic Regression Model or a Deep Neural Network Classifier depending on need

    if model == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns)
    elif model == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[25, 10], n_classes=num_classes)
    elif model == "both":
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[25, 10],
            n_classes=num_classes)

    return m


def input_func(df):
    # Create a dictionary mapping from each continuous feature column name (k) to the values of that column stored in a constant Tensor.

    continuous_cols = {k: tf.constant(df[k].values)
                       for k in continuous_columns}

    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.

    categorical_cols = {k: tf.SparseTensor(
        indices = [[i, 0] for i in range(df[k].size)],
        values = df[k].values,
        shape = [df[k].size, 1])
                        for k in categorical_columns}

    # Merge two dictionaries into one
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)

    # Converts the label column into a constant Tensor with dtype tf.int64
    label = tf.constant(df["label"].values)
    label = tf.to_int64(label)

    # Returns the feature columns and the label
    return feature_cols, label



def train_and_evaluate(model_call):
    # With this function, we will be parsing through tweets & getting
    # the tokenized words of the tweet (while throwing out
    # stop words?)
    # For each of the tweets we will determine if
    # positive apply, then predict the current category
    # based on previous categories 
    # Now to train and evaluate model
    train_file = "train.csv"
    test_file = "test.csv"

    df_train = pd.read_csv(train_file, names = train_columns)
    df_test = pd.read_csv(test_file, names = test_columns)
    print("Shape of Training Set: ")
    print(df_train.shape)
    print("Shape of Test Set: ")
    print(df_test.shape)

    print("Columns in Training Set: ")
    print(df_train.columns)
    print("Columns in Test Set: ")
    print(df_test.columns)

    model_dir = tempfile.mkdtemp()

    m = categorize(model_dir, model_call)
    print("Model categorized")

    m.fit(input_fn = lambda: input_func(df_train), steps = 200)
    print("Model fitted")

    results = m.evaluate(input_fn = lambda: input_func(df_test), steps = 1)
    for key in sorted(results):
        print("{}: {}".format(key, results[key]))


    df_unlabeled = pd.read_csv("unlabeled.csv", names = ['id', 'word', 'label'])

    print(df_unlabeled.shape)
    print(df_unlabeled.columns)

    y = m.predict(input_fn = lambda: input_func(df_unlabeled))
    print ('Predictions: {}'.format(str(y)))


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
    wordslist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features
"""
