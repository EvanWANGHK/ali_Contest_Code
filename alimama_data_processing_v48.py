#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:50:39 2018

@author: zhaolicheng
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import scipy.special as special
from sklearn.metrics import log_loss
from sklearn import preprocessing
import warnings
import math
warnings.filterwarnings("ignore")

import time

data = pd.concat((pd.read_csv("feature_file.txt"),\
                  pd.read_csv("feature_file_v2.txt"),\
                  pd.read_csv("feature_file_v5.txt")),axis = 1)

def sub(train, test):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp','day','item_id','user_id','flag'
                     ]]
    X = train[col]
    y = train['is_trade'].values
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
        reg_alpha=10,
        objective='binary',
        # metric='binary_error',
        num_leaves=15,
        max_depth=5,
        learning_rate=0.05,
#        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=3000)#500
    lgb_model = lgb0.fit(X, y)
#    predictors = [i for i in X.columns]
#    feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['predicted_score'] = pred
    sub1 = test[['instance_id', 'predicted_score']]
    sub=pd.read_csv('round2_ijcai_18_test_b_20180510.txt', sep=" ")
    sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
    sub=sub.fillna(0)
    sub[['instance_id', 'predicted_score']].to_csv('predict_20180514_1.txt',sep=" ",index=False)

train= data[(data['is_trade'] >= 0) & (data['day'] >= 4)& (data['day'] <= 7)]
test = data[data.is_trade.isnull()]
sub(train, test)