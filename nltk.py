import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords')

pos_tweets = []
neg_tweets = []

all_tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [word.lower() for word in words.split() if len(word) >= 3] 
    tweets.append((words_filtered, sentiment))
