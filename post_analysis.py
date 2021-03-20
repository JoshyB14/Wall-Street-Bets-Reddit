# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Wall Street Bets (WSB) sub-reddit post analysis
# 
# [Wall Street Bets](https://www.reddit.com/r/wallstreetbets/) according to [Wikipedia](https://en.wikipedia.org/wiki/R/wallstreetbets) is:
# 
#  > A subreddit where participants discuss stock and option trading. It has become notable for its profane nature, aggressive trading strategies, and role in the GameStop short squeeze that caused losses on short positions in U.S. firms topping US$70 billion in a few days in early 2021. The subreddit is famous for its colorful jargon and terms.
# 
# As mentioned from Wiki, we will examine the short squeeze that occured from around the 12th of Jan 2021 to ~~the 4th of Feb 2021~~ the 18th of March 2021 on Gamestop (NYSE: GME) and AMC Entertainment Holdings (NYSE: AMC). 
# 
# For more information see [this Bloomberg article](https://www.bloomberg.com/news/features/2021-02-04/gamestop-gme-how-wallstreetbets-and-robinhood-created-bonkers-stock-market) and [this blog post](https://www.wallstreetbets.shop/blogs/news/dissecting-the-unique-lingo-and-terminology-used-in-the-subreddit-r-wallstreetbets) from the WSB offical merch shop on terms/ emojis used on the sub-reddit.
# 
# 
#  

# %%
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style("darkgrid")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Import regular expressions package (regex)
import re


# %%
wsb = pd.read_csv('data/reddit_wsb.csv')
wsb.head()


# %%
wsb.info()

# %% [markdown]
# The data has 8 columns that represent the following (Taken from [Python Reddit API Wrapper](https://github.com/reddit-archive/reddit/wiki/JSON)):
# 
# - title: the title of the link. may contain newlines for some reason
# 
# - score: the net-score of the link. note: A submission's score is simply the number of upvotes minus the number of downvotes. If five users like the submission and three users don't it will have a score of 2. Please note that the vote numbers are not "real" numbers, they have been "fuzzed" to prevent spam bots etc. So taking the above example, if five users upvoted the submission, and three users downvote it, the upvote/downvote numbers may say 23 upvotes and 21 downvotes, or 12 upvotes, and 10 downvotes. The points score is correct, but the vote totals are "fuzzed".
# 
# - id: this item's identifier, e.g. "8xwlg"
# 
# - url: the link of this post. the permalink if this is a self-post
# 
# - comms_num: the number of comments that belong to this link. includes removed comments.
# 
# - created: the time of creation in local epoch-second format
# 
# - body: the raw text. this is the unformatted text which includes the raw markup characters such as ** for bold. <, >, and & are escaped.
# 
# - timestamp: datetime about the related activity
# %% [markdown]
# Encode timestamp as date and extract date elements.

# %%
wsb['timestamp'] = pd.to_datetime(wsb['timestamp'])
wsb['date'] = pd.to_datetime(wsb['timestamp']).dt.date
wsb['date'] = pd.to_datetime(wsb['date'])
wsb['weekday'] = pd.to_datetime(wsb['timestamp']).dt.weekday
wsb['hour'] = pd.to_datetime(wsb['timestamp']).dt.hour


# %%
# Sanity check
wsb.info()


# %%
# Examine missing data
for column in wsb.columns:
    print(f'Column {column}', f'has {100 * sum(wsb[column].isnull())/len(wsb):.2f}% missing data')
    print()

# %% [markdown]
# From the code above we see that the only column with missing values is the 'body' column with ~48% missing data.

# %%
wsb.describe()


# %%
# Drop columns that are not needed
wsb = wsb.drop(columns=['id','url','created'])

# %% [markdown]
# Before going any further we will define some of the terms/words/abbreviations that we may see in these posts:
# - GME --> The ticker code for Gamestop
# - AMC --> Ticker code for AMC Entertainment Holdings
# - Robinhood (RH) --> A US brokerage firm offering 'free' trading on ETFs and equities. Millienial focused and is very popular with Gen Y and Z. [See here](https://robinhood.com/us/en/)
# - NOK --> Ticker code for Nokia. This stock was also a target of the WSB community
# 
# For more info see [this blog post](https://www.wallstreetbets.shop/blogs/news/dissecting-the-unique-lingo-and-terminology-used-in-the-subreddit-r-wallstreetbets).
# 
# 

# %%
date_min = wsb['date'].min()
date_max = wsb['date'].max()
print(f'Starting date in data: {date_min}')
print(f'Most recent date in data: {date_max}')

# %% [markdown]
# Remove 2020-09-29 00:00:00 as it its outside the 2021 Gamestop saga

# %%
wsb = wsb[wsb['date']!= date_min]


# %%
# Sanity check
date_min = wsb['date'].min()
date_max = wsb['date'].max()
print(f'Starting date in data: {date_min}')
print(f'Most recent date in data: {date_max}')


# %%
sns.histplot(x='date', data=wsb)
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# A majority of posts occured on or around the 29th of Jan. Lets see what day and time of week the posts were created.
# 
# <img src='images/calender.gif'>
# 
# [Source](https://tinyurl.com/ygnv9gyk)

# %%
# .index will take 0-6 (days of week), .values (takes number of posts per day of week)
# Eg. .values gives [18818  4642  3282  3247  2009  1737  1375]
sns.barplot(x=wsb['weekday'].value_counts().index, y=wsb['weekday'].value_counts().values)
plt.title('WSB posts by day of week')
plt.ylabel('Num of posts')
plt.xlabel('Day of week')
plt.xticks([0,1,2,3,4,5,6],['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.show()


# %%
sns.barplot(x=wsb['hour'].value_counts().index, y=wsb['hour'].value_counts().values)
plt.title('WSB posts by hour of day')
plt.ylabel('Num of posts')
plt.xlabel('Hour of day')
plt.show()

# %% [markdown]
# Interestingly a majority of posts were posted on a Friday, at the end of a trading week. However looking further into the hour of the posts, posted we see that a majority of posts were posted arounf 1am-3am. (For reference, the timezone is UTC +8, AKA Pacific Standard Time (PST), which covers most the western US and Canada. Source: https://en.wikipedia.org/wiki/Pacific_Time_Zone)

# %%
# Add 'posts' column to sum over posts by date
wsb['post_count'] = 1
by_date = wsb.groupby('date').aggregate({'score': 'mean', 'comms_num': 'mean', 'post_count': 'sum'}).reset_index()
# Change date to string for plotting purposes
by_date['date'] = by_date['date'].astype('string')
# Sanity check
by_date.head()


# %%
plt.figure(figsize=(20,10))
sns.barplot(x='date', y='score', data=by_date)
plt.xticks(rotation=90)
plt.title('Average reddit score by date')
plt.plot()


# %%
plt.figure(figsize=(20,10))
sns.barplot(x='date', y='comms_num', data=by_date)
plt.xticks(rotation=90)
plt.title('Average number of comments by date')
plt.show()


# %%
plt.figure(figsize=(20,10))
sns.barplot(x='date', y='post_count', data=by_date)
plt.xticks(rotation=90)
plt.title('Number of posts by date')
plt.show()

# %% [markdown]
# ## Zooming in on key dates

# %%
hype1 = wsb[(wsb['date']>='2021-1-29')& (wsb['date']<='2021-2-2')]
# Change date to string for plotting purposes
hype1['date'] = hype1['date'].astype('string')


# %%
sns.barplot(x='date', y='score', data=hype1, ci=None)
plt.xticks(rotation=45)
plt.title('Total reddit score of posts over the weekend of 29th Jan - 2nd Feb')
plt.show()


# %%
hype2 = wsb[wsb['date']<='2021-2-24']
# Change date to string for plotting purposes
hype2['date'] = hype2['date'].astype('string')


# %%
plt.figure(figsize=(20,10))
sns.barplot(x='date', y='score', data=hype2, ci=None)
plt.xticks(rotation=45)
plt.title('Total reddit score of posts from 24th Feb onwards')
plt.show()

# %% [markdown]
# From the chart below of the GME stock price (and volume) [sourced from Yahoo Finance](https://tinyurl.com/yzdb6bbe):
# 
# 1. When GME hit an all time high of $380 USD (closing price) on the 28th of Jan - we see a surge in posts on the 29th of Jan, up to just below 16,000 posts
# 
# 2. Around the same time above, the number of comments increases to around 500 on the 30th of Jan (Saturday - [Source](https://tinyurl.com/yehvwjv4)), then drops to around 300 comments on the 31st of Jan (Sunday), before rising again to around 500 comments per day on the 1st of Feb (Monday)
# 
# 3. After a major drop in share price, with a subsequent rise in price from the beginning of March onwards - we see a small rise in post count on a handful of days in March
# 
# 4. The sum of the score of reddit posts (a mertic of engagement on a post/ posts) incresed after the inital GME runup to $380 USD (29th Jan), then a reasonably steady rise/ increase in scores from the second runup of GME from the 24th Feb onwards.
# 
# <img src='images/GME_chart.png'>
# 
# %% [markdown]
# ## Sentiment Analysis

# %%
# Combine title and body reddit post text
# Create function to combine and clean text
def clean_combine(text):
    # convert all text to lowercase
    text = text.lower()
    # Replace handlers with empty string
    text = re.sub('@[^\s]+','', text)
    # Replace any links (URLS)
    text = re.sub(r"http\S+", '', text)
    # Replace all special characters
    text = ' '.join(re.findall(r'\w+', text))
    # Replace single characters with an empty string
    text = re.sub(r'\s+[a-zA-Z]\s+', '', text)
    # Replace any extra whitespace
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text


# Text preprocessing
wsb_title = wsb['title'].dropna()
wsb_body = wsb['body'].dropna()
wsb_title = wsb_title.apply(lambda x: clean_combine(x))
wsb_body = wsb_body.apply(lambda x: clean_combine(x))


# %%
print(wsb_title)


