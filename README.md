# sentimentAnalysis_presCandidatesUSA2016

ABSTRACT

This project focuses on classifying tweets (source: [twitter](https://twitter.com)) of the 2016 candidates for presidency to the United States: Hillary Clinton (Democrat) and Donald Trump (Republican) into two classes: positive or negative. Natural language processing (NLP) is used in the context of the Python programming language -- we focus on leveraging the [NLTK package](http://www.nltk.org/). 
This project is written in Python

ADDITIONAL RESOURCES

We used the following documentation to further educate ourselves through the process of this project.


 - [Larent Luce's Blog : Twitter sentiment analysis using Python and NLTK](http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/)

 - This tutorial builds tweetAnalysis.py

 - [Tensorflow Software via Google : Deep Learning Algorithms](https://www.tensorflow.org/)

  - This tutorial is used to build nltk

 - [Mining Twitter with Python - by CodeKitchen](http://web.mit.edu/aizhan/www/twitter_api_workshop/#/)

  - This tutorial is used to build file tweetsByHandle.py, handles.csv, and handles.db

  - [The Art of Tokenization](https://www.ibm.com/developerworks/community/blogs/nlp/entry/tokenization?lang=en)

  - This will give you an idea on the methods used for tokenizing text (we used "low-level tokenization" for this repo)

DEPENDENCIES REQUIRED

The following dependences (in Python) are required to run this program:

    nltk 
    csv
    random
    numpy
    pandas
    time
    sys (optional)
    sklearn

STEPS REQUIRED

[Convert the .db file into a csv file](http://stackoverflow.com/questions/3286525/return-sql-table-as-json-in-python)

Use the this code below:

    python tweetAnalysis.py parse
    python tweetAnalysis.py train
    python tweetAnalysis.py compare
    

METHOD

 - Scrape data from Twitter

 - Output the data as a database file

 - Using SQL software (MySQL, DB Browser for SQLite, etc.), extract the various columns which you are doing analysis on

 - Look up SQL syntax if you are not familiar [here](

 - Gather the necessary columns in this order ("row_id", "tweet_id", "timestamp", "president", "tweet", "label") and save the output as "tweets.csv"

 - You can also extract other data and in a different order; just make sure to change the column data at the top of tweetAnalysis.py

 - Now run the code above in the following order (or use the lovely Makefile, which does it all for you!)


GENERAL FINDINGS 

It took Hillary 11 days to get to 200 tweets and Trump 16 days to get to 200 tweets

The amount of positive tweets stayed relatively the same over the course of the week; negative and neutral tweets were all over the place

Hillary's tweets were generally more neutral overall while Trump's were highly polarizing (positive or negative)