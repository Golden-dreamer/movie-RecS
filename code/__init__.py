#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:18:06 2022

@author: leo
"""
import pandas as pd
import numpy as np

import Recommender


PATH_MOVIES = "../data/ml-latest-small/movies.csv"
PATH_TAGS = "../data/ml-latest-small/tags.csv"
PATH_RATINGS = "../data/ml-latest-small/ratings.csv"
PATH_ME = '../data/my_ratings.csv'


def getData():
    ''' return 3 dataFrames: movies, tags, ratings.'''
    movies = pd.read_csv(PATH_MOVIES, usecols=['movieId',
                                               'title', 'genres'])
    tags = pd.read_csv(PATH_TAGS, usecols=['userId',
                                           'movieId', 'tag'])
    ratings = pd.read_csv(PATH_RATINGS, usecols=['userId',
                                                 'movieId', 'rating'])
    movies.columns = ['itemId', 'title', 'genres']
    tags.columns = ['userId', 'itemId', 'tag']
    ratings.columns = ['userId', 'itemId', 'rating']
    return movies, tags, ratings


# Normalize data
def getNormDf(df):
    return df.apply(lambda row: row / np.sqrt(df[df != 0].count(axis=1)))


def getMe():
    cols = ['movieId', 'rating']
    me = pd.read_csv(PATH_ME, usecols=cols)
    me.columns = ['itemId', 'rating']
    return me.dropna()


def addMeToRating(me, ratings):
    maxUserId = ratings.userId.max()
    me['userId'] = maxUserId + 1
    return pd.concat([ratings, me])


# This function should be in other place
def getUserProfile(user, df):
    '''return user Profile based on users rates'''
    return pd.Series(np.dot(user, df), index=df.columns,
                     name=user.name, dtype=float)


def dataframeWithTermFreq(dataframe):
    '''return df contains frequency, How may times tag applied to the item,
    index=itemId, columns = tags'''
    tf = dataframe.groupby(['itemId', 'tag']).count()
    tf.columns = ['freq']
    tf = tf.astype(int)
    pivot = tf.reset_index().pivot(columns='tag', index='itemId',
                                   values='freq')

    return pivot


items, tags, ratings = getData()
me = getMe()  # also gives userId to me DataFrame
ratingsFull = addMeToRating(me, ratings)

MY_USER_ID = me.userId.max()  # можно оптимизировать наносекунду(нет)

tags.tag = tags.tag.str.lower()

recs = Recommender.Recommender(ratingsFull, items, tags)
recommends = recs.predict(MY_USER_ID)

# df = tags.tag.value_counts()  # document frequency
# # Could be 1 / df
# # idf = 1/df
# totalNumberOfItemsWithTag = tags.movieId.nunique()
# idf = np.log(totalNumberOfItemsWithTag / df)


# # tf[tf.index.get_level_values('tag') == 'funny']
# tf = dataframeWithTermFreq(tags)
# #tf = tf.join(items.set_index('movieId'), how='outer').drop(
# #                                               columns=['genres', 'title'])
# tf = tf.fillna(0)
# tf = getNormDf(tf)
# # can add genres as tag in future
# tf = tf.join(items.set_index('itemId'), how='outer').drop(
#                                               columns=['genres', 'title'])
# tf = tf.fillna(0)
# tf = tf.sort_index()


# userRatings = ratingsFull[ratingsFull.userId == MY_USER_ID]
# userRatings = userRatings.merge(items.movieId, how='outer')
# userRatings = userRatings.set_index('itemId').drop(columns='userId').rating
# userRatings = userRatings.fillna(0)
# userRatings = userRatings.sort_index()

# userProfile = getUserProfile(userRatings, tf)


# def getRecs(tf, idf, userProfile, userRatings, itens, dropWatched=True):
#     watched = userRatings[userRatings != 0].index
#     recs = (idf * tf * userProfile).sum(axis=1)
#     recs = recs.sort_values(ascending=False)
#     recs.name = 'predScore'
#     fullInfoRecs = items.set_index('itemId').join(recs)
#     if dropWatched:
#         fullInfoRecs = fullInfoRecs.drop(index=watched)
#     fullInfoRecs = fullInfoRecs[fullInfoRecs.predScore != 0]
#     fullInfoRecs = fullInfoRecs.sort_values('predScore', ascending=False)
#     return fullInfoRecs


# recs = getRecs(tf, idf, userProfile, userRatings, items)
