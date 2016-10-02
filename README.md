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

  - This will give you an idea on the methods used for tokenizing text (we used "low-level tokenization" for this repo)

  - [The Art of Tokenization](https://www.ibm.com/developerworks/community/blogs/nlp/entry/tokenization?lang=en)


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

First run the following program:
      
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

and you will get the same result as the three python commands above

## METHOD

 - Scrape data from Twitter using the "twitterScraper.py" function

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