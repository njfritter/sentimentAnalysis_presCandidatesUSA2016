# This simple piece of code will allow us to prompt the user for
# predictions of sentiment for tweets and then compare them to the
# predictions of the algorithm
import csv
import tweetAnalysis

in_file = "predicted.csv"
out_file = "classified_tweets.csv"

tweets_file = csv.reader(open(in_file, "rb"))
classified_tweets = csv.writer(open(out_file, "wb"))

index = 0
correct = 0

for tweet in tweets_file:
        (president, tweet, predicted) = row
        print(tweet)
        sentiment = raw_input("Enter your sentiment: ")
        classified_tweets.writerow([president] + [tweet] + [predicted] + [sentiment])
        if str(predicted) == str(sentiment):
            correct += 1
        index += 1

"""
for tweet in classified_tweets:    
    (president, tweet, predicted, sentiment) = row
    if predicted == sentiment:
        correct += 1
    index += 1
"""

print("The percent similarity our guesses were to the algorithm were: %d" % ((float(correct / index)) * 100))
