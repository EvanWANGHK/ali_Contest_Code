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
#%%

def timestamp_datetime(value):
    fmt = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(fmt, value)
    return dt

def base_process(data):
    lbl = preprocessing.LabelEncoder()
    
    for i in range(1, 3):
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].\
             map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))            
    for i in range(10):
        data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].\
             map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for col in ['item_id', 'item_brand_id', 'item_city_id','user_id','shop_id']:
        data[col] = lbl.fit_transform(data[col])
    
    data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['day'] = data['realtime'].dt.day
    data[data['day']==31] = 0
    data['hour'] = data['realtime'].dt.hour
    
    data['minute'] = data['realtime'].dt.minute
    data['second'] = data['realtime'].dt.second
    data['hour_min_sec'] = data['hour'] + data['minute']/60.0 + data['second']/3600.0
    data['hour_min_sec_sin'] = data['hour_min_sec'].map(lambda x: math.sin((x-12)/24*2*math.pi))
    data['hour_min_sec_cos'] = data['hour_min_sec'].map(lambda x: math.cos((x-12)/24*2*math.pi))
    
    del data['minute'],data['second'],data['hour_min_sec']

    for i in range(10):
        data['predict_category_property' + str(i)] = \
        lbl.fit_transform(data['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    
    return data

def slide_cnt_sub1(data):
    for length in range(1,8):
        for d in range(0+length, 8):  
            df1 = data[data['day'] == d - length] 
            df2 = data[data['day'] == d]  
            user_cnt = df1.groupby(by='user_id').size().to_dict()
            item_cnt = df1.groupby(by='item_id').size().to_dict()
            shop_cnt = df1.groupby(by='shop_id').size().to_dict()
            df2['user_cnt_%d' %length] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
            df2['item_cnt_%d' %length] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
            df2['shop_cnt_%d' %length] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
            df2 = df2[['user_cnt_%d' %length, 'item_cnt_%d' %length,
                       'shop_cnt_%d' %length, 'instance_id']]
            if d == 0+length:
                Df2 = df2
            else:
                Df2 = pd.concat([df2, Df2])
        data = pd.merge(data, Df2, on=['instance_id'], how='left')
    return data

def slide_cnt_sub2(data):    
    for d in range(1, 8):       
        df1 = data[data['day'] < d]
        df2 = data[data['day'] == d]
        user_cnt = df1.groupby(by='user_id').size().to_dict()
        item_cnt = df1.groupby(by='item_id').size().to_dict()
        shop_cnt = df1.groupby(by='shop_id').size().to_dict()
        df2['user_cnt_all'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cnt_all'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cnt_all'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cnt_all', 'item_cnt_all', 'shop_cnt_all', 'instance_id']]
        if d == 1:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')
    return data

def slide_cnt(data):  
    data = slide_cnt_sub1(data)
    data = slide_cnt_sub2(data)
    return data
#%%
def difftime(data):    
    usergroup = data[['user_id','instance_id','context_timestamp']].groupby('user_id')
    th = 1e4
    for x in usergroup:        
        dataslice = x[1].drop(labels = ['user_id'],axis = 1).sort_values('context_timestamp')
        dataslice["user_time_diff_1"] = np.insert(np.diff(dataslice["context_timestamp"]).\
                 astype(float),0,np.nan)
        dataslice["user_time_diff_2"] = np.insert(np.diff(dataslice["context_timestamp"],2).\
                 astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),2))
        dataslice["user_time_diff_3"] = np.insert(np.diff(dataslice["context_timestamp"],3).\
                 astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),3))
        dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True)   
        try:
            diff = pd.concat([diff,dataslice])           
            if len(diff)>th:
                print(len(diff))
                th += 1e4
        except:
            diff = dataslice.copy()
    data = pd.merge(data, diff, on=['instance_id'], how='left') 
    del diff
    
    itemgroup = data[['item_id','instance_id','context_timestamp']].groupby('item_id')
    th = 1e4
    for x in itemgroup:        
        dataslice = x[1].drop(labels = ['item_id'],axis = 1).sort_values('context_timestamp')
        dataslice["item_time_diff_1"] = np.insert(np.diff(dataslice["context_timestamp"]).\
                 astype(float),0,np.nan)
        dataslice["item_time_diff_2"] = np.insert(np.diff(dataslice["context_timestamp"],2).\
                 astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),2))
        dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True)   
        try:
            diff = pd.concat([diff,dataslice])           
            if len(diff)>th:
                print(len(diff))
                th += 1e4
        except:
            diff = dataslice.copy()
    data = pd.merge(data, diff, on=['instance_id'], how='left') 
    del diff 
    
    shopgroup = data[['shop_id','instance_id','context_timestamp']].groupby('shop_id')
    th = 1e4
    for x in shopgroup:        
        dataslice = x[1].drop(labels = ['shop_id'],axis = 1).sort_values('context_timestamp')
        dataslice["shop_time_diff_1"] = np.insert(np.diff(dataslice["context_timestamp"]).\
                 astype(float),0,np.nan)
        dataslice["shop_time_diff_2"] = np.insert(np.diff(dataslice["context_timestamp"],2).\
                 astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),2))
        dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True)   
        try:
            diff = pd.concat([diff,dataslice])           
            if len(diff)>th:
                print(len(diff))
                th += 1e4
        except:
            diff = dataslice.copy()
    data = pd.merge(data, diff, on=['instance_id'], how='left') 
    del diff
    return data
#%%
class BayesianSmoothing(object):
    def __init__(self, alpha = 1, beta = 1):
        self.alpha = alpha
        self.beta = beta

    def update1(self, imps, clks, iter_num = 50, epsilon = 0.005):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            print("iter: %d" %i , "alpha_change: %.4f" %abs(new_alpha - self.alpha), \
                  " beta_change: %.4f" %abs(new_beta - self.beta))
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                self.alpha = new_alpha
                self.beta = new_beta
                break
            else:
                self.alpha = new_alpha
                self.beta = new_beta
            

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        
        numerator_alpha = (special.digamma(clks + alpha) - special.digamma(alpha)).sum()
        numerator_beta = (special.digamma(imps - clks + beta) - special.digamma(beta)).sum()
        denominator = (special.digamma(imps + alpha + beta) - special.digamma(alpha + beta)).sum()

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)

    def update2(self, imps, clks):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(imps, clks)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, imps, clks):
        '''moment estimation'''
            
        ctr_list = [float(x)/y if y != 0 else 0 for (x,y) in zip(clks,imps)]           
        mean = sum(ctr_list)/len(ctr_list)
        ctr_list_2 = [(x-mean)*(x-mean) for x in ctr_list]
        var = sum(ctr_list_2)/(len(ctr_list_2)-1)
        return mean, var
#%%
def obj_sub(data,x):
    objcnt = data.groupby([x], as_index=False)['flag'].\
    agg({x+'_cnt': 'sum'})
    data = pd.merge(data, objcnt, on=[x], how='left')
    return data

def cross_obj(data,l1,l2):
    for x in l1:
        for y in l2:
            cross_obj_cnt = data.groupby([x, y], as_index=False)['flag'].\
            agg({x + '_' + y + '_cnt': 'sum'})
            data = pd.merge(data, cross_obj_cnt, on=[x, y], how='left')
            bs = BayesianSmoothing()
            bs.update2(data[x+'_cnt'],data[x + '_'+ y + '_cnt'])           
            print("alpha: %.4f" %bs.alpha , "beta: %.4f" %bs.beta)
            data[x + '_'+ y + '_prob']= (data[x + '_'+ y + '_cnt'] + bs.alpha)/ \
            (data[x+'_cnt'] + bs.alpha + bs.beta)
            del data[x + '_'+ y + '_cnt']
    return data

def obj_time(data,l):
    for x in l:
        obj_query_day = data.groupby([x, 'day'], as_index=False)["flag"].\
        agg({x + '_query_day':'sum'})
        data = pd.merge(data, obj_query_day, 'left', on=[x, 'day'])
        obj_query_day_hour = data.groupby([x, 'day', 'hour'], as_index=False)["flag"].\
        agg({x + '_query_day_hour':'sum'})
        data = pd.merge(data, obj_query_day_hour, 'left',
                        on=[x, 'day', 'hour'])
    return data

def obj(data,l):
    for x in l:
        data = obj_sub(data,x)
    return data

def item(data):
    l = ['item_id','item_brand_id',
              ]
    data = obj_time(data,l)   
    
    l = ['item_id','item_brand_id','item_city_id',
              'item_price_level','item_sales_level','item_collected_level',
              'item_pv_level']
    data = obj(data,l)
       
    l1 = ['item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level',
              'item_pv_level']
    l2 = ['item_id',]
    data = cross_obj(data,l1,l2)
    
    l1 = ['item_city_id','item_price_level','item_sales_level','item_collected_level',
              'item_pv_level']
    l2 = ['item_brand_id',]
    data = cross_obj(data,l1,l2)
                
    return data

def user(data):
    l = ['user_id','user_gender_id','user_age_level', 
                'user_occupation_id',]
    data = obj_time(data,l) 

    
    l = ['user_id','user_gender_id','user_age_level', 
                'user_occupation_id', 'user_star_level']
    data = obj(data,l)
    
    l1 = ['user_gender_id','user_age_level', 
                'user_occupation_id', 'user_star_level']
    l2 = ['user_id']
    data = cross_obj(data,l1,l2)
    
    l1 = ['user_gender_id', 
                ]
    l2 = ['user_age_level','user_occupation_id', 'user_star_level']
    data = cross_obj(data,l1,l2)

    return data

def shop(data):
    l = ['shop_id',
                ]
    data = obj_time(data,l) 
    
    l = ['shop_id','shop_review_num_level','shop_star_level', 
                ]
    
    data = obj(data,l)
    
    l1 = ['shop_review_num_level','shop_star_level',]
    l2 = ['shop_id']
    data = cross_obj(data,l1,l2)

    return data

def context(data):
    l = ['context_page_id']
    data = obj_time(data,l)
    
    l = ['context_page_id']
    for x in l:
        data = obj_sub(data,x)
        
    return data
    

def user_item(data):
    l1 = ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level',
                'context_page_id'
                ]
    l2 = ['user_id',]
    
    data = cross_obj(data,l1,l2)
    
    l1 = ['user_gender_id','user_age_level', 
                'user_occupation_id', 'user_star_level'
                ]
    l2 = ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level',
                ]
    
    data = cross_obj(data,l1,l2)
    
    return data


def user_shop(data):
    l1 = ['shop_id','shop_review_num_level', 'shop_star_level',
                ]
    l2 = ['user_id',               
                ]    
    data = cross_obj(data,l1,l2)
    
    l1 = ['user_gender_id','user_age_level', 
                'user_occupation_id', 'user_star_level',
                'context_page_id'
                ]
    l2 = ['shop_id',               
                ]    
    data = cross_obj(data,l1,l2)

    return data


def shop_item(data):   
    l1 = ['shop_id','shop_review_num_level', 'shop_star_level',
                'context_page_id'
                ]
    l2 = ['item_id']
                
    data = cross_obj(data,l1,l2)
    
    l1 = ['shop_id',
                ]
    l2 = ['item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']
    data = cross_obj(data,l1,l2)
    return data

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
        reg_alpha = 10,#0
        objective='binary',
        # metric='binary_error',
        num_leaves=35,#7
        max_depth=6,#3
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
        reg_alpha = 10,#0
        objective='binary',
        # metric='binary_error',
        num_leaves=35,#7
        max_depth=6,#3
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
    sub=pd.read_csv("round2_ijcai_18_test_a_20180425.txt", sep=" ")
    sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
    sub=sub.fillna(0)
    sub[['instance_id', 'predicted_score']].to_csv('predict_20180426_1.txt',sep=" ",index=False)
#%%
if __name__ == "__main__":
    train = pd.read_csv("round2_train.txt", sep=" ")
    print('train data load done')
    test = pd.read_csv("round2_ijcai_18_test_a_20180425.txt", sep=" ")
    print('test data load done')
    data = pd.concat([train, test])
    del train,test
    data["flag"] = 1
#%%
    data = base_process(data)
    print('base done')
#%%
    data = slide_cnt(data)  
    print('slide_cnt done')
#%%
#    data = difftime(data) 
#    print('difftime done')  
#%%
    data = item(data)   
    data = user(data)
    data = shop(data)
    data = context(data)
    data = user_item(data)
    data = user_shop(data)
    data = shop_item(data)
#%%
#    dellabel = []
#    l = [
#              'item_price_level','item_sales_level','item_collected_level',
#              'item_pv_level'] + ['user_gender_id','user_age_level', 
#                'user_occupation_id', 'user_star_level'] + ['shop_review_num_level','shop_star_level', 
#                ] + ['context_page_id']
#    dellabel += map(lambda x: x+'_cnt',l)
#    data.drop(labels = dellabel,axis = 1,inplace = True)
    data.to_csv("feature_file.txt",index = False)
#%%    
    train= data[(data['day'] >= 0) & (data['day'] <= 5)]   
    test= data[(data['day'] == 6)]  
#%%
    best_iter,feat_imp = lgbCV(train, test)
#%%    
    del train,test
    train = data[data.is_trade.notnull()]  
    test = data[data.is_trade.isnull()]  
    sub(train, test, best_iter)