#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:17:01 2022

@author: leo
"""
import pandas as pd
import numpy as np


def getNormDf(df):
    return df.apply(lambda row: row / np.sqrt(df[df != 0].count(axis=1)))


class Recommender:
    def __init__(self, ratings=None, items=None, tags=None):
        self.ratings = ratings
        self.items = items
        self.tags = tags
        self.tf = self.dataframeWithTermFreq(tags).fillna(0)
        self.tf = self.normAddMissedFillNaNSort()
        df = tags.tag.value_counts()  # document frequency
        # Could be 1 / df
        # idf = 1/df
        totalNumberOfItemsWithTag = tags.itemId.nunique()
        self.idf = np.log(totalNumberOfItemsWithTag / df)

    def addMissedItems(self, df):
        return df.join(self.items.set_index('itemId'), how='outer').drop(
                                                columns=['genres', 'title'])

    def normAddMissedFillNaNSort(self, normalize=True):
        tf = self.tf
        if normalize:
            tf = getNormDf(self.tf)
        # can add genres as tag in future
        tf = self.addMissedItems(tf)
        tf = tf.fillna(0)
        tf = tf.sort_index()
        return tf

    def dataframeWithTermFreq(self, dataframe):
        '''return df contains frequency, How may times tag applied to the item,
        index=itemId, columns = tags'''
        tf = dataframe.groupby(['itemId', 'tag']).count()
        tf.columns = ['freq']
        tf = tf.astype(int)
        pivot = tf.reset_index().pivot(columns='tag', index='itemId',
                                       values='freq')

        return pivot

    def getUserProfile(self, userId):
        userRatings = self.ratings[self.ratings.userId == userId].merge(
                                        self.items.itemId, how='outer')
        userRatings = userRatings.set_index('itemId').sort_index().rating.fillna(0)
        self.userRatings = userRatings
        '''return user Profile based on users rates'''
        return pd.Series(np.dot(userRatings, self.tf), index=self.tf.columns,
                         name=userRatings.name, dtype=float)

    def predict(self, userId, userProfile=None, dropWatched=True):
        if userProfile is None:
            userProfile = self.getUserProfile(userId)
        self.userProfile = userProfile
        watched = self.userRatings[self.userRatings != 0].index
        recs = (self.idf * self.tf * userProfile).sum(axis=1)
        recs = recs.sort_values(ascending=False)
        recs.name = 'predScore'
        fullInfoRecs = self.items.set_index('itemId').join(recs)
        if dropWatched:
            fullInfoRecs = fullInfoRecs.drop(index=watched)
        fullInfoRecs = fullInfoRecs[fullInfoRecs.predScore != 0]
        fullInfoRecs = fullInfoRecs.sort_values('predScore', ascending=False)
        return fullInfoRecs
