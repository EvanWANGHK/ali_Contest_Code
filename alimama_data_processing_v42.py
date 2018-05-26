import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Manager,Pool

userlists = Manager().list()

def usertimediff(x):
    dataslice = x[1].drop(labels = ['user_id'],axis = 1).sort_values('context_timestamp')
    dataslice["user_time_diff_1"] = np.insert(np.diff(dataslice["context_timestamp"]).\
             astype(float),0,np.nan)
    dataslice["user_time_diff_2"] = np.insert(np.diff(dataslice["context_timestamp"],2).\
             astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),2))
    dataslice["user_time_diff_3"] = np.insert(np.diff(dataslice["context_timestamp"],3).\
             astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),3))
    dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True)  
    userlists.append(dataslice)

itemlists = Manager().list()

def itemtimediff(x):
    dataslice = x[1].drop(labels = ['item_id'],axis = 1).sort_values('context_timestamp')
    dataslice["item_time_diff_1"] = np.insert(np.diff(dataslice["context_timestamp"]).\
             astype(float),0,np.nan)
    dataslice["item_time_diff_2"] = np.insert(np.diff(dataslice["context_timestamp"],2).\
             astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),2))
    dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True)  
    itemlists.append(dataslice)
    
shoplists = Manager().list()  
  
def shoptimediff(x):
    dataslice = x[1].drop(labels = ['shop_id'],axis = 1).sort_values('context_timestamp')
    dataslice["shop_time_diff_1"] = np.insert(np.diff(dataslice["context_timestamp"]).\
             astype(float),0,np.nan)
    dataslice["shop_time_diff_2"] = np.insert(np.diff(dataslice["context_timestamp"],2).\
             astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),2))
    dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True) 
    shoplists.append(dataslice)   


def difftime_parallel(data):    
    usergroup = data[['user_id','instance_id','context_timestamp']].groupby('user_id')
    pool=Pool()
    global userlists
    for x in usergroup:  
        pool.apply_async(usertimediff,args=(x,)) 
    pool.close()
    pool.join()
    feat = pd.concat(userlists).reset_index(drop = True)
    del userlists
    data = pd.merge(data, feat, on=['instance_id'], how='left') 
    del feat

    itemgroup = data[['item_id','instance_id','context_timestamp']].groupby('item_id')
    pool=Pool()
    global itemlists
    for x in itemgroup:  
        pool.apply_async(itemtimediff,args=(x,)) 
    pool.close()
    pool.join()
    feat = pd.concat(itemlists).reset_index(drop = True)
    del itemlists
    data = pd.merge(data, feat, on=['instance_id'], how='left') 
    del feat
    
    shopgroup = data[['shop_id','instance_id','context_timestamp']].groupby('shop_id')
    pool=Pool()
    global shoplists
    for x in shopgroup:  
        pool.apply_async(shoptimediff,args=(x,)) 
    pool.close()
    pool.join()
    feat = pd.concat(shoplists).reset_index(drop = True)    
    del shoplists
    data = pd.merge(data, feat, on=['instance_id'], how='left') 
    del feat
    
    return data
#%%
if __name__ == "__main__":
    usecol = ['user_id','item_id','shop_id','instance_id','context_timestamp']
    train = pd.read_csv("round2_train.txt", sep=" ",usecols = usecol)
    print('train data load done')
    testa = pd.read_csv("round2_ijcai_18_test_a_20180425.txt", sep=" ",usecols = usecol)
    testb = pd.read_csv("round2_ijcai_18_test_b_20180510.txt", sep=" ",usecols = usecol)
    print('test data load done')
    data = pd.concat([train, testa, testb])
    del train,testa,testb
    dellabel = data.columns
#%%
    data = difftime_parallel(data) 

    data.drop(labels = dellabel,axis = 1,inplace = True)
    data.to_csv("feature_file_v2.txt",index = False)