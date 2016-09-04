# This simple piece of code will allow us to prompt the user for
# predictions of sentiment for tweets and then compare them to the
# predictions of the algorithm
import csv
import tweetAnalysis

file = "predicted.csv"


tweets_file = csv.reader(open(file, "rb"))
classified_tweets = csv.writer(open("classified_tweets.csv", "wb"))

for tweet in tweets_file:
        (president, tweet, predicted) = row
        print(tweet)
        sentiment = raw_input("Enter your sentiment: ")
        
        classified_tweets.writerow([president] + [tweet] + [predicted] + [sentiment])
        
index = 0
correct = 0

for tweet in classified_tweets:    
    (president, tweet, predicted, sentiment) = row
    if predicted == sentiment:
        correct += 1
    index += 1

print("The percent similarity our guesses were to the algorithm were: %d" % ((float(correct / index)) * 100))
