# Here for this project we will be analyzing Tweets about the
# Presidential Candidates by word, tokenizing them, and then using
# them to classify each overall tweet as either positive or negative

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')

# A set of positive and negative tweets
pos_tweets = []
neg_tweets = []

# Now let's combine all the tweets
all_tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [word.lower() for word in words.split() if len(word) >= 3] 
    tweets.append((words_filtered, sentiment))

# Now the tweets are all 
