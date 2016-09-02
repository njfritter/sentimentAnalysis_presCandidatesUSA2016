#!/usr/bin/env python

import sys
import string
import simplejson
from twython import Twython

# we will use the variables day, month, year for output filename
import datetime
now = datetime.datetime.now()
day = int(now.day)
month = int(now.month)
year = int(now.year)

# for oauth authentication -- needed to access the twitter api
t = Twython(app_key = 'nHL6RdyZuqPAJ4VSrK48pUQCU',
	    app_secret = 'urwTOqQAVsTXCGu0tixUSxEs2zSfuZ5ptAVa2nA9qjQYi6cuG4',
	    oauth_token = '261933490-bNxEiTOkx89bsqCwgFxZVKxLwt8J3PYoOtrDVLlE',
	    oauth_token_secret = 'igONbEDQ1ohEekc5IqYJlrdeUWACKZBVKCo5XPwWP6Q88')

# declaring twitter id's such that
	# HILLARY CLINTON'S TWITTER INFO
		#	Twitter User ID:	1339835893
		#	Full Name:			Hillary Clinton
		#	Screen Name:		HillaryClinton
		#	Total Followers:	8,557,582
		#	Total Statuses:		7,770
	# DONALD TRUMP'S TWITTER INFO
		#	Twitter User ID:	25073877
		#	Full Name:			Donald J. Trump
		#	Screen Name:		realDonaldTrump
		#	Total Followers:	11,262,784
		#	Total Statuses:		33,075
ids = "1339835893, 25073877"

# run the lookup_user method within the twitter api
# grab info on up to 100 ids with each api call
# the variable users is a JSON file with data on the 32 twitter users listed above
users = t.lookup_user(user_id = ids)

# name the output file with %i getting replaced by month, day, year
outfn = "twitter_user_data_%i.%i.%i.txt" % (now.month, now.day, now.year)

# names for header row in output file
fields = "id screen_name name created_at url followers_count friends_count statuses_count \
		  favorites_count listed_count \
		  contributors_enabled description protected location lang expanded_url".split()

# initialize the output file and write header row
outfp = open(outfn, "w")
outfp.write(string.join(fields, "\t") + "\n") # header

#THE VARIABLE 'USERS' CONTAINS INFORMATION OF THE 32 TWITTER USER IDS LISTED ABOVE
#THIS BLOCK WILL LOOP OVER EACH OF THESE IDS, CREATE VARIABLES, AND OUTPUT TO FILE
for entry in users:
    #CREATE EMPTY DICTIONARY
    r = {}
    for f in fields:
        r[f] = ""
    #ASSIGN VALUE OF 'ID' FIELD IN JSON TO 'ID' FIELD IN OUR DICTIONARY
    r['id'] = entry['id']
    #SAME WITH 'SCREEN_NAME' HERE, AND FOR REST OF THE VARIABLES
    r['screen_name'] = entry['screen_name']
    r['name'] = entry['name']
    r['created_at'] = entry['created_at']
    r['url'] = entry['url']
    r['followers_count'] = entry['followers_count']
    r['friends_count'] = entry['friends_count']
    r['statuses_count'] = entry['statuses_count']
    r['favourites_count'] = entry['favourites_count']
    r['listed_count'] = entry['listed_count']
    r['contributors_enabled'] = entry['contributors_enabled']
    r['description'] = entry['description']
    r['protected'] = entry['protected']
    r['location'] = entry['location']
    r['lang'] = entry['lang']
    #NOT EVERY ID WILL HAVE A 'URL' KEY, SO CHECK FOR ITS EXISTENCE WITH IF CLAUSE
    if 'url' in entry['entities']:
        r['expanded_url'] = entry['entities']['url']['urls'][0]['expanded_url']
    else:
        r['expanded_url'] = ''
    print r
    #CREATE EMPTY LIST
    lst = []
    #ADD DATA FOR EACH VARIABLE
    for f in fields:
        lst.append(unicode(r[f]).replace("\/", "/"))
    #WRITE ROW WITH DATA IN LIST
    outfp.write(string.join(lst, "\t").encode("utf-8") + "\n")

outfp.close()    