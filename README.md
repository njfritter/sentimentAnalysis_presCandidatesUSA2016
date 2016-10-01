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

First run the following program:
      
      python twitterScraper.py

This function will extract raw tweets using the Twitter API from the presidential 
candidates, Donald Trump and Hillary Clinton. The tweet limit is currently set in 
line 177; feel free to change this if you like (there is a tweet limit however).

[Convert the .db file into a csv file](http://stackoverflow.com/questions/3286525/return-sql-table-as-json-in-python)

You need to download a SQL engine (such as DB for SQLite) in order to parse the .db file and turn it into a csv

Once in there, run the following SQL command:

     SELECT rowid, tweet_id, created_at, query, content, possibly_sensitive FROM output

Once you do this, go and find the option to save the parse as a csv file (follow 
instructions on how to do this depending on the SQL software you have). Save this file
as "tweets.csv", then run the following code for analysis:

    python tweetAnalysis.py parse
    python tweetAnalysis.py train
    python tweetAnalysis.py compare
    
There is also a "Makefile" with the same commands as above. You may also run:

      make

and you will get the same result. 

METHOD

 - Scrape data from Twitter using the "twitterScraper.py" function

 - You will get a database (.db) file as "handles.db"

 - Using SQL software (MySQL, DB Browser for SQLite, etc.), look for "handles.db" and extract the various columns which you are doing analysis on

 - Look up SQL syntax if you are not familiar [here](

 - Running the SQL command above will generate the desired features

 - The SQL command generates these features ("row_id", "tweet_id", "", "president", "tweet", "label") and save the output as "tweets.csv"

 - You can also extract other data and in a different order; just make sure to change the "columns" data at the top of tweetAnalysis.py

 - Now run the code above in the following order (or use the lovely Makefile, which does it all for you!)


GENERAL FINDINGS 

It took Hillary 11 days to get to 200 tweets and Trump 16 days to get to 200 tweets

The amount of positive tweets stayed relatively the same over the course of the week; negative and neutral tweets were all over the place

Hillary's tweets were generally more neutral overall while Trump's were highly polarizing (positive or negative)