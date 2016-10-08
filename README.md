# Sentiment Analysis on Presidential Candidates for election year 2016

## ABSTRACT

This project focuses on classifying tweets (source: [twitter](https://twitter.com)) of the 2016 candidates for presidency to the United States: Hillary Clinton (Democrat) and Donald Trump (Republican) into two classes: positive or negative. Natural language processing (NLP) is used in the context of the Python programming language -- we focus on leveraging the [NLTK package](http://www.nltk.org/). 
This project is written in Python, and a translation to R is in the works.

## ADDITIONAL RESOURCES

We used the following documentation to further educate ourselves through the process of this project.

 - This tutorial was used to build tweetAnalysis.py (the one we will be using)

 - [Larent Luce's Blog : Twitter sentiment analysis using Python and NLTK](http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/)

 - This tutorial was used to build tensorflow.py (has yet to work)

 - [Tensorflow Software via Google : Deep Learning Algorithms](https://www.tensorflow.org/)

  - This tutorial was used to build twitterScraper.py, tweetsByHandle.py, handles.csv, and handles.db

 - [Mining Twitter with Python - by CodeKitchen](http://web.mit.edu/aizhan/www/twitter_api_workshop/#/)

## PICKING THE ALGORITHM

Sentiment Analysis has been a common topic of discussion ever since the explosion of data analytics onto the world, but how exactly can you take a tweet and determine whether it is positive or negative? How would one do this? Through a process called Tokenization. This process takes sentences as input and "tokenizes" them into individual words. This will give you an idea on the methods used for tokenizing text (we used "low-level tokenization" for this repo) and could explain it better than this repo could:

[The Art of Tokenization](https://www.ibm.com/developerworks/community/blogs/nlp/entry/tokenization?lang=en)

I hand labeled the first data set myself (leading to a labeled set with my biases), but have found a corpus of sentences labeled as positive and negative that
I will be using to relabel the sentences and compare. This will be coming soon. The link to the corpus of words can be found [here](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)

## DEPENDENCIES REQUIRED

The following dependences (in Python) are required to run tweetAnalysis.py and twitterScraper.py:

    nltk 
    csv
    random
    numpy
    pandas
    time
    sys (optional)
    sklearn
    matplotlib
    twython
    sys
    string
    sqlite3
    pprint
    simplejson
    sqlalchemy

	
## STEPS REQUIRED

### Creating a Twitter Application

This section is for those of you reading this that do not currently have a Twitter application with a set of API and Oauth keys. If you do have one already, please continue to the next section.

If you do not already have a Twitter Application to scrape tweets from the Twitter API, please visit their (Applications Website)[https://apps.twitter.com/] and create a profile. It is recommended that you have a website that you can use as the "home" website but it is not necessary. 

Once you have created your profile, click on "Keys and Access Tokens" and get the first two keys (Consumer Key and Consumer Secret). Keep these for the next step.

First create a file called "api_keys.csv" with your keys in the following order:
consumer key      
consumer secret key
access token
secret access token

Keep this file private (do NOT commit to Github). 

Next run the following program:
      
      python twitterScraper.py

This function will extract raw tweets using the Twitter API from the presidential candidates, Donald Trump and Hillary Clinton. The tweet limit is currently set in line 177; feel free to change this if you like (there is a tweet limit however).

You need to download a SQL engine (such as DB for SQLite) in order to parse the .db file and turn it into a csv. Here is the link for SQLite below:

- [SQLite Download Link](https://www.sqlite.org/download.html)

Assuming you are using SQLite, open the database file "handles.db", click in "Browse Data", switch the table to "output" and run the following SQL command:

     SELECT rowid, tweet_id, created_at, query, content, possibly_sensitive FROM output

Once you do this, go and find the option to save the parse as a csv file (follow instructions on how to do this depending on the SQL software you have). For SQLite this option will be below whatever outputs from the above SQL command; save this file as "tweets.csv", then run the following code for analysis:

    python tweetAnalysis.py parse
    python tweetAnalysis.py train
    python tweetAnalysis.py compare
    
There is also a "Makefile" with the same commands as above. You may also run:

      make

and the three python commands above will run in that order.

## METHOD

 - Scrape data from Twitter using the "twitterScraper.py" function and 

 - You will get a database (.db) file as "handles.db"

 - Using SQL software (MySQL, DB Browser for SQLite, etc.), look for "handles.db" and extract the various columns which you are doing analysis on

 - Look up SQL syntax if you are not familiar [here](http://www.w3schools.com/sql/)

 - Running the SQL command in the previous section will generate the desired features

 - The SQL command generates these features ("row_id", "tweet_id", "", "president", "tweet", "label") and save the output as "tweets.csv"

 - You can also extract other data and in a different order; just make sure to change the "columns" data at the top of tweetAnalysis.py

 - Now run the code above in the following order (or use the lovely Makefile, which does it all for you!)


## GENERAL FINDINGS 

It took Hillary 11 days to get to 200 tweets and Trump 16 days to get to 200 tweets

The amount of positive tweets stayed relatively the same over the course of the week; negative and neutral tweets were all over the place

Hillary's tweets were generally more neutral overall while Trump's were highly polarizing (positive or negative)