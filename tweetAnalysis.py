# Here for this project we will be analyzing Tweets about the
# Presidential Candidates by word, tokenizing them, and then using
# them to classify each overall tweet as either positive or negative
# For the tweets here I have already classified 300 out of the 400
# tweets as either positive, negative or neutral
# This will allow the algorithm to be trained on 200 data points 
# And tested on a test set of the remaining 100 data points
# 100 of the data points are unlabeled and predicted with the algorithm
# Here we will import the Scikit Learn libraries among others necessary

import pandas as pd
import csv
import sys
import random
import numpy as np
import time
import twython
from time import strftime
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import nltk
from nltk import corpus
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.twitter import Twitter
 
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2

#from pandas.DataFrame import query

# Here are the files we will be manipulating and creating
file = "tweets.csv"
randomized_file = "randomized_tweets.csv"
training_data = "training_data.csv"
testing_data = "testing_data.csv"
unlabeled_data = "unlabeled.csv"
predicted_data_NB = "predicted_nb.csv"
predicted_data_LSVM = "predicted_lsvm.csv"
predicted_data_MLP = "predicted_mlp.csv"
amazon_reviews = "sentimentLabelledSentences/amazonLabelled.csv"
imdb_reviews = "sentimentLabelledSentences/imdbLabelled.csv"
yelp_reviews = "sentimentLabelledSentences/yelpLabelled.csv"
randomized_amazon_file = "randomized_amazon_reviews.csv"
randomized_imdb_file = "randomized_imdb_reviews.csv"
randomized_yelp_file = "randomized_yelp_reviews.csv"
amazon_training_data = "amazon_training_data.csv"
amazon_testing_data = "amazon_testing_data.csv"
imdb_training_data = "imdb_training_data.csv"
imdb_testing_data = "imdb_testing_data.csv"
yelp_training_data = "yelp_training_data.csv"
yelp_testing_data = "yelp_testing_data.csv"


# Define columns for my labelled data sets
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
    "month",
    "day",
    "hour",
    "president",
    "tweet",
    "label"
    ]

test_columns = [
    "row_id",
    "tweet_id",
    "month",
    "day",
    "hour",
    "president",
    "tweet",
    "label"
    ]

unlabeled_columns = [
    "row_id",
    "tweet_id",
    "month"
    "day",
    "hour",
    "president",
    "tweet",
    "label"
    ]

# Label columns in researcher labelled stuff
research_columns = [
    "row",
    "review", 
    "label"
    ]

labels = []

local_stopwords = []

pos_freq = np.zeros(7)
neg_freq = np.zeros(7)
neu_freq = np.zeros(7)

pos_freq_trump = np.zeros(7)
neg_freq_trump = np.zeros(7)
neu_freq_trump = np.zeros(7)

pos_freq_clinton = np.zeros(7)
neg_freq_clinton = np.zeros(7)
neu_freq_clinton = np.zeros(7)

"""
def get_tweets():
    tw = Twitter()
    
    # Sample from the public stream based on keyword
    tw.tweets(keywords='love, hate', limit=10) 
    
    # See what Donald Trump and Hillary Clinton are talking about
    # respectively 
    # Use numeric userIDs instead of handles
    
    tw.tweets(follow=['25073877', '1339835893'], limit=10) 
"""

def parse_csv():
    # Here we will parse a CSV file with the data on Row ID, Tweet ID,
    # Timestamp, President, Tweet

    training_file = csv.writer(open(training_data, "wt+"))
    testing_file = csv.writer(open(testing_data, "wt+"))
    unlabeled_file = csv.writer(open(unlabeled_data, "wt+"))
        
    # Now to randomize the data; this is how
    # Gotten from Github: 
    # (http://stackoverflow.com/questions/4618298/randomly-mix-lines-of-3-million-line-file)
    with open(file, 'rt') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(randomized_file, 'wt+') as target:
        for _, line in data:
            target.write( line )
    
    prepped_tweet_file = csv.reader(open(randomized_file, "rt"))
    #prepped_tweet_file = open(randomized_file, "rt", encoding = "utf-8">)
    index = 0

    # Now we will iterate through the randomized file and extract data
    # We need to get rid of the decimal points in the seconds columns
    # And then split up the data (2/3 train and 1/3 test)
    # And obtain frequencies as well
    
    for row in prepped_tweet_file:
        (row_id, tweet_id, timestamp, president, tweet, label) = row
        raw_timestamp = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
        ratio = 3
        
        # Take care of unlabeled data
        if label == "1":
            tokenize_row_write(unlabeled_file, row_id, tweet_id, raw_timestamp.tm_mon, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, "")
            continue
 
   
        # Get frequencies of sentiment per day of week
        # Index values of array used for day of week
        if label == "positive":
            pos_freq[raw_timestamp.tm_wday] += 1
        if label == "negative":
            neg_freq[raw_timestamp.tm_wday] += 1
        else:
            neu_freq[raw_timestamp.tm_wday] += 1
            
        # Now do this for each candidate
        if president == "HillaryClinton":
            if label == "positive":
                pos_freq_clinton[raw_timestamp.tm_wday] += 1
            if label == "negative":
                neg_freq_clinton[raw_timestamp.tm_wday] += 1
            else:
                neu_freq_clinton[raw_timestamp.tm_wday] += 1

        if president == "realDonaldTrump":
            if label == "positive":
                pos_freq_trump[raw_timestamp.tm_wday] += 1
            if label == "negative":
                neg_freq_trump[raw_timestamp.tm_wday] += 1
            else:
                neu_freq_trump[raw_timestamp.tm_wday] += 1

   
        # Before we write our tweets to the files, let's remove all hyperlinks 
        # Gotten from this url http://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
        # Maybe here too??
        # http://billchambers.me/tutorials/2015/01/14/python-nlp-cheatsheet-nltk-scikit-learn.html
        tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)

        # Now write to files
        if index % ratio == 0:
            tokenize_row_write(testing_file, row_id, tweet_id, raw_timestamp.tm_mon, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, label)
        else:
            tokenize_row_write(training_file, row_id, tweet_id, raw_timestamp.tm_mon, raw_timestamp.tm_wday, raw_timestamp.tm_hour, president, tweet, label)
            
        index += 1
    
    # Create data visualizations of
    # Overall tweet sentiment by day of week
    # Hillary's tweet sentiment by day of week    
    # Trump's tweet sentiment by day of week    
    data_viz(pos_freq, neu_freq, neg_freq, "")
    data_viz(pos_freq_clinton, neu_freq_clinton, neg_freq_clinton, "Hillary Clinton")
    data_viz(pos_freq_trump, neu_freq_trump, neg_freq_trump, "Donald Trump")

def clean_word(word):
    return word not in stopwords.words('english') and word not in local_stopwords

def tokenized_string(sent):
    tokenizer = RegexpTokenizer(r'\w+')
    words = [word.lower() for word in tokenizer.tokenize(sent)]
    words = [word for word in words if clean_word(word)]
    
    return words

# Tokenize the title and description, then write everything to the corresponding
# csv file
def tokenize_row_write(file_csv_writer, row_id, tweet_id, month, day, hour, president, tweet, label):
    words_tweet = tokenized_string(tweet)

    file_csv_writer.writerow([row_id] + [tweet_id] + [month] + [day] + [hour] + [president]+ [words_tweet] + [label])

def data_viz(pos_freq, neu_freq, neg_freq, president):

    if president == "":
        print("Positive frequencies by day of the week are: ")
        print(pos_freq)
        print("Neutral frequencies by day of the week are: ")
        print(neu_freq)
        print("Negative frequencies by day of the week are: ")
        print(neg_freq)

    else:
        print("Positive frequencies by day of the week for candidate {} are: ".format(president))
        print(pos_freq)
        print("Neutral frequencies by day of the week for candidate {} are: ".format(president))
        print(neu_freq)
        print("Negative frequencies by day of the candidate {} week are: ".format(president))
        print(neg_freq)
        
    #Plot a multibar graph 
    numbers = np.arange(0, 7)
    w = 0.2
    
    ax = plt.subplot(111)
    ax.bar(numbers - w, pos_freq, width = w, color = 'b', align = 'center')
    ax.bar(numbers, neu_freq, width = w, color = 'g', align = 'center')
    ax.bar(numbers + w, neg_freq, width = w, color = 'r', align = 'center')
    ax.autoscale(tight=True)

    # Add the axis labels
    ax.set_ylabel("Frequencies of Sentiment")
    ax.set_xlabel("Day of Week")
       
    # Add title if the president argument is used
    if len(president) > 0:
        ax.set_title('Tweet Sentiment of Presidential Candidate {} by Day of the Week'.format(president))    
    else:
        ax.set_title('Tweet Sentiment of Both Presidential Candidates by Day of the Week')

    # Add legend
    blue_patch = mpatches.Patch(color = 'blue', label = 'Positive Tweets')
    green_patch = mpatches.Patch(color = 'green', label = 'Neutral Tweets')
    red_patch = mpatches.Patch(color = 'red', label = 'Negative Tweets')
    ax.legend(handles = [blue_patch, green_patch, red_patch], loc = 9)

    plt.show() 
 
def extract_and_train():
    # Train based on my labeled dataset (which does contain my bias)
    train = pd.read_csv(training_data, names = train_columns)
    test = pd.read_csv(testing_data, names = test_columns)
    
    x_train = np.array((train['row_id'], train['tweet_id'], train['month'], train['day'], train['hour'], train['president'], train['tweet']))
    y_train = np.array(train['label'])
    
    x_test = np.array((test['row_id'], test['tweet_id'], test['month'], test['day'], test['hour'], test['president'], test['tweet']))
    y_test = np.array(test['label'])
    
    train_words = np.array(train['tweet'])
    test_words = np.array(test['tweet'])
    
    # Show unique labels
    unique, counts = np.unique(y_train[0], return_counts=True)
    print(np.asarray((unique, counts)).T)

    # Extract features from text files
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_words)
    print(X_train_counts.shape)
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)
    
    # Training a classifer
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    test_sentence = ['Our plan promises to revive the economy and build jobs. Trump will destroy the middle class and feed his pockets']
    X_new_counts = count_vect.transform(test_sentence)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    # Make a prediction of test sentence
    predicted = clf.predict(X_new_tfidf)
    for words, category in zip(test_words, predicted):
        print('%r => %s' % (words, category))

    # Call a Naive Bayes, a Linear SVM and a Neural Network algorithm on the data
    naive_bayes(train_words, y_train, test_words, y_test)
    linear_svm(train_words, y_train, test_words, y_test)
    neural_network(train_words, y_train, test_words, y_test)

def research_split():
    # Train based on the researcher labeled corpus
    # Which will be more objective than mine
    # Files to be read in
    """
    amazon_file = pd.read_csv(amazon_reviews, names = research_columns)
    imdb_file = pd.read_csv(imdb_reviews, names = research_columns)
    yelp_file = pd.read_csv(yelp_reviews, names = research_columns)
    """
    # Some of the reviews only contained numbers with a "NA" label
    # Adding in code to skip over those lines
    # Also had to do some MAJOR data reformatting
    # Link used here: 
    # http://stackoverflow.com/questions/32127604/how-to-wrap-each-line-in-quotes-in-sublimetext
    amazon_file = csv.reader(open(amazon_reviews, "rt+"))
    imdb_file = csv.reader(open(imdb_reviews, "rt+"))
    yelp_file = csv.reader(open(yelp_reviews, "rt+"))

    # Files to be written
    amazon_train_file = csv.writer(open(amazon_training_data, "wt+"))
    amazon_test_file = csv.writer(open(amazon_testing_data, "wt+"))
    imdb_train_file = csv.writer(open(imdb_training_data, "wt+"))
    imdb_test_file = csv.writer(open(imdb_testing_data, "wt+"))
    yelp_train_file = csv.writer(open(yelp_training_data, "wt+"))
    yelp_test_file = csv.writer(open(yelp_testing_data, "wt+"))

    """
    # Randomize data
    with open(amazon_file, 'rb') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(randomized_amazon_file, 'wt+') as target:
        for _, line in data:
            target.write( line )

    with open(imdb_file, 'rb') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(randomized_imdb_file, 'wt+') as target:
        for _, line in data:
            target.write( line )

    with open(yelp_file, 'rb') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(randomized_yelp_file, 'wt+') as target:
        for _, line in data:
            target.write( line )
    
    print("Files randomized")
    prepped_amazon_file = csv.reader(open(randomized_amazon_file, "rb"))
    prepped_imdb_file = csv.reader(open(randomized_imdb_file, "rb"))
    prepped__file = csv.reader(open(randomized_imdb_file, "rb"))
    
    """

    # replace files with "prepped" versions when possible
    index = 0
    for row in amazon_file:
        (number, review, label) = row
        ratio = 3

        if index % ratio == 0:
            tokenize_row_write_research(amazon_test_file, number, review, label)
        else:
            tokenize_row_write_research(amazon_train_file, number, review, label)
            
        index += 1

    index = 0
    for row in imdb_file:
        (number, review, label) = row
        ratio = 3

        if index % ratio == 0:
            tokenize_row_write_research(imdb_test_file, number, review, label)
        else:
            tokenize_row_write_research(imdb_train_file, number, review, label)
            
        index += 1

    index = 0
    for row in yelp_file:
        (number, review, label) = row
        ratio = 3

        if index % ratio == 0:
            tokenize_row_write_research(yelp_test_file, number, review, label)
        else:
            tokenize_row_write_research(yelp_train_file, number, review, label)
            
        index += 1

def research_train():

    amazon_train = pd.read_csv(amazon_training_data, names = research_columns)
    amazon_test = pd.read_csv(amazon_testing_data, names = research_columns)
    imdb_train = pd.read_csv(imdb_training_data, names = research_columns)
    imdb_test = pd.read_csv(imdb_testing_data, names = research_columns)
    yelp_train = pd.read_csv(yelp_training_data, names = research_columns)
    yelp_test = pd.read_csv(yelp_testing_data, names = research_columns)

    amazon_x_train = np.array((amazon_train['row'], ['review']))
    amazon_y_train = np.array(amazon_train['label'])

    amazon_x_test = np.array((amazon_test['row'], ['review']))
    amazon_y_test = np.array(amazon_test['label'])

    imdb_x_train = np.array((imdb_train['row'], ['review']))
    imdb_y_train = np.array(imdb_train['label'])

    imdb_x_test = np.array((imdb_test['row'], ['review']))
    imdb_y_test = np.array(imdb_test['label'])

    yelp_x_train = np.array((yelp_train['row'], ['review']))
    yelp_y_train = np.array(yelp_train['label'])

    yelp_x_test = np.array((yelp_test['row'], ['review']))
    yelp_y_test = np.array(yelp_test['label'])

    amazon_train_words = np.array(amazon_train['review'])
    amazon_test_words = np.array(amazon_test['review'])

    imdb_train_words = np.array(imdb_train['review'])
    imdb_test_words = np.array(imdb_test['review'])

    yelp_train_words = np.array(yelp_train['review'])
    yelp_test_words = np.array(yelp_test['review'])

    # Run functions on all three corpuses
    print("\nThe results using Amazon reviews are:")
    naive_bayes(amazon_train_words, amazon_y_train, amazon_test_words, amazon_y_test)
    linear_svm(amazon_train_words, amazon_y_train, amazon_test_words, amazon_y_test)
    #neural_network(amazon_train_words, amazon_y_train, amazon_test_words, amazon_y_test)
    
    print("\nThe results using IMDB reviews are:")
    naive_bayes(imdb_train_words, imdb_y_train, imdb_test_words, imdb_y_test)
    linear_svm(imdb_train_words, imdb_y_train, imdb_test_words, imdb_y_test)
    #neural_network(amazon_train_words, amazon_y_train, amazon_test_words, amazon_y_test)

    print("\nThe results using Yelp reviews are:")
    naive_bayes(yelp_train_words, yelp_y_train, yelp_test_words, yelp_y_test)
    linear_svm(yelp_train_words, yelp_y_train, yelp_test_words, yelp_y_test)
    #neural_network(amazon_train_words, amazon_y_train, amazon_test_words, amazon_y_test)

def tokenize_row_write_research(file_csv_writer, number, review, label):
    
    words_review = tokenized_string(review)
    file_csv_writer.writerow([number] + [words_review] + [label])

def naive_bayes(x_train, y_train, x_test, y_test):
    # Building a Pipeline; this does all of the work in extract_and_train() at once  
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('chi2', SelectKBest(chi2, k = 'all')),
                         ('clf', MultinomialNB()),
                         ])
 
    text_clf = text_clf.fit(x_train, y_train)
    
    # Evaluate performance on test set
    predicted = text_clf.predict(x_test)
    print("The accuracy of a Naive Bayes algorithm is: ") 
    print(np.mean(predicted == y_test))
    print("Number of mislabeled points out of a total %d points for the Naive Bayes algorithm : %d"
          % (x_test.shape[0],(y_test != predicted).sum()))
    
    # Tune parameters and predict unlabeled tweets
    parameter_tuning(text_clf, x_train, y_train)
    predict_unlabeled_tweets(text_clf, predicted_data_NB)

def linear_svm(x_train, y_train, x_test, y_test):
    """ Let's try a Linear Support Vector Machine (SVM) """

    text_clf_two = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('chi2', SelectKBest(chi2, k = 'all')),
                             ('clf', SGDClassifier(
                                 loss='hinge',
                                 penalty='l2',
                                 alpha=1e-3,
                                 n_iter=5,
                                 random_state=42)),
                            ])
    text_clf_two = text_clf_two.fit(x_train, y_train)
    predicted_two = text_clf_two.predict(x_test)
    print("The accuracy of a Linear SVM is: ")
    print(np.mean(predicted_two == y_test))
    print("Number of mislabeled points out of a total %d points for the Linear SVM algorithm: %d"
          % (x_test.shape[0],(y_test != predicted_two).sum()))
    
    # Tune parameters and predict unlabeled tweets
    parameter_tuning(text_clf_two, x_train, y_train)
    predict_unlabeled_tweets(text_clf_two, predicted_data_LSVM)

def neural_network(x_train, y_train, x_test, y_test):
    """ Now let's try one of the most powerful algorithms: An Artifical Neural Network (ANN) """

    text_clf_three = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('chi2', SelectKBest(chi2, k = 'all')),
                                ('clf', MLPClassifier(
                                        hidden_layer_sizes=(100,), 
                                        max_iter=10, 
                                        alpha=1e-4,
                                        solver='sgd', 
                                        verbose=10, 
                                        tol=1e-4, 
                                        random_state=1,
                                        learning_rate_init=.1)),
                                ])

    text_clf_three.fit(x_train, y_train)
    print("Training set score: %f" % text_clf_three.score(x_train, y_train))
    print("Test set score: %f" % text_clf_three.score(x_test, y_test))

    """
    fig, axes = plt.subplots(4, 4)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = text_clf_three.coefs_[0].min(), text_clf_three.coefs_[0].max()
    for coef, ax in zip(text_clf_three.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), 
                    cmap=plt.cm.gray, 
                    vmin=.5 * vmin,
                    vmax=.5 * vmax)

        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()
    """
    
    # Tune parameters and predict unlabeled tweets
    parameter_tuning(text_clf_three, x_train, y_train)
    predict_unlabeled_tweets(text_clf_three, predicted_data_MLP)

def parameter_tuning(text_clf, x_train, y_train):
    """ 
    Classifiers can have many different parameters that can make the                                                                                                             
    algorithm more accurate (MultinomialNB() has a smoothing                                                                                                                            parameter, SGDClassifier has a penalty parameter, etc.). Here we                                                                                                                    will run an exhaustive list of the best possible parameter values 
    """
    
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
        
        """                                                                                                                                                                                 print(metrics.classification_report(y_test, predicted,                                                                                                                                 
        target_names=twenty_test.target_names))       
        """

def predict_unlabeled_tweets(classifier, output):
    # Make predictions
    unlabeled_tweets = pd.read_csv(unlabeled_data, names = unlabeled_columns)
    unlabeled_words = np.array(unlabeled_tweets["tweet"])
    predictions = classifier.predict(unlabeled_words)
    print(predictions)
    
    # Create new file for predictions
    # And utilize csv module to iterate through csv
    predicted_tweets = csv.writer(open(output, "wt+"))
    unlabeled_tweets = csv.reader(open(unlabeled_data, "rt+"))
    
    # Iterate through csv and get president and tweet
    # Add prediction to end
    # Also recieved from Github:
    # http://stackoverflow.com/questions/23682236/add-a-new-column-to-an-existing-csv-file
    index = 0
    for row in unlabeled_tweets:
        (row_id, tweet_id, month, day, hour, president, tweet, label) = row
        predicted_tweets.writerow([president] + [tweet] + [predictions[index]])
        index += 1

def compare_predictions():
    # Compare the predictions of both algorithms
    names = ["president", "tweet", "prediction"]
    naive_bayes = pd.read_csv(predicted_data_NB, names = names) 
    linear_svm = pd.read_csv(predicted_data_LSVM, names = names) 

    naive_bayes_pred = np.array(naive_bayes["prediction"])
    linear_svm_pred = np.array(linear_svm["prediction"])
    
    print("The precent similarity between a Multinomial Naive Bayes Algorithm and a Linear SVM algorithm with a SGD Classifier is: ")
    print(np.mean(naive_bayes_pred == linear_svm_pred))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'tweets':
            get_tweets()
        elif sys.argv[1] == 'parse':
            parse_csv()
        elif sys.argv[1] == 'train':
            extract_and_train()
        elif sys.argv[1] == 'compare':
            compare_predictions()
    if len(sys.argv) == 3:
        if sys.argv[1] == 'research':
            if sys.argv[2] == 'split':
                research_split()
            if sys.argv[2] == 'train':
                research_train()







