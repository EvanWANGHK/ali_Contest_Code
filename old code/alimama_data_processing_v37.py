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

#data = pd.concat((pd.read_csv("feature_file.txt"),\
#                  pd.read_csv("feature_file_v2.txt"),\
#                  pd.read_csv("feature_file_v4.txt")),axis = 1)
#data = pd.concat((pd.read_csv("feature_file.txt"),\
#                  pd.read_csv("feature_file_v2.txt"),\
#                  pd.read_csv("feature_file_v5.txt")),axis = 1)
data = pd.concat((pd.read_csv("feature_file_v3.txt"),\
                  pd.read_csv("feature_file_v2.txt"),\
                  pd.read_csv("feature_file_v4.txt"),\
                  pd.read_csv("feature_file_v5.txt")),axis = 1)
#%%
#train= data[(data['day'] >= 0) & (data['day'] <= 4)]   
#test= data[(data['day'] == 5)] 
#%% 
def lgbCV(train, test):
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp','day','item_id','user_id','flag',
                     ]]
    # cat = ['sale_price', 'gender_star', 'user_age_level', 'item_price_level', 'item_sales_level', 'sale_collect',
    #        'price_collect', 'item_brand_id', 'user_star_level', 'item_id', 'shop_id',
    #        'item_city_id', 'context_page_id', 'gender_age', 'shop_star_level', 'item_pv_level', 'user_occupation_id',
    #        'day', 'gender_occ', 'user_gender_id']
    X = train[col]
    y = train['is_trade'].values
    X_tes = test[col]
    y_tes = test['is_trade'].values

    lgb0 = lgb.LGBMClassifier(
        reg_alpha=10,
        objective='binary',
        # metric='binary_error',
        num_leaves=35,
        max_depth=6,
        learning_rate=0.05,
#        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=200000)
    lgb_model = lgb0.fit(X, y, eval_set=[(X_tes, y_tes)], early_stopping_rounds=200,verbose = False)
    best_iter = lgb_model.best_iteration_
    predictors = [i for i in X.columns]
    feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
    print(feat_imp.shape)
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['pred'] = pred
    # print(test[['is_trade','pred']])
    print('error: ', log_loss(test['is_trade'], test['pred']))
    return best_iter,feat_imp
#best_iter,feat_imp = lgbCV(train, test)
#%%
def sub(train, test, best_iter):
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
        num_leaves=35,
        max_depth=6,
        learning_rate=0.05,
#        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=500)
    lgb_model = lgb0.fit(X, y)
#    predictors = [i for i in X.columns]
#    feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['predicted_score'] = pred
    sub1 = test[['instance_id', 'predicted_score']]
    sub=pd.read_csv('round2_ijcai_18_test_a_20180425.txt', sep=" ")
    sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
    sub=sub.fillna(0)
    sub[['instance_id', 'predicted_score']].to_csv('predict_20180506_3.txt',sep=" ",index=False)

#del train,test
train = data[data.is_trade.notnull()].copy()
#train= data[(data.is_trade.notnull()) & (data['day'] >= 4)& (data['day'] <= 7)].copy()  
test = data[data.is_trade.isnull()].copy()  
del data
#%%
#sub(train, test, best_iter) 
sub(train, test, 0)