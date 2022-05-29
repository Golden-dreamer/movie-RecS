#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:18:06 2022

@author: leo
"""
import pandas as pd
import numpy as np

import Recommender


PATH_MOVIES = "../data/ml-latest/movies.csv"
PATH_TAGS = "../data/ml-latest/tags.csv"
PATH_RATINGS = "../data/ml-latest/ratings.csv"
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


def reduceUsers(DF, userThreshold=20):  # 20
    df = DF.copy()
    df = df.groupby('userId').count()
    users = df.loc[df.rating > userThreshold].index
    return DF.loc[DF.userId.isin(users)]


items, tags, ratings = getData()
me = getMe()  # also gives userId to me DataFrame
ratingsFull = reduceUsers(addMeToRating(me, ratings))
#ratingsFull = addMeToRating(me, ratings)

MY_USER_ID = me.userId.max()  # можно оптимизировать наносекунду(нет)
tags.tag = tags.tag.str.lower()

recs = Recommender.Recommender(ratingsFull, items, tags, freqThreshold=20)
recommends = recs.predict(MY_USER_ID, dropWatched=True)
