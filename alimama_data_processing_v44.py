import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Manager,Pool

useritemlists = Manager().list()

def useritemtimediff(x):
    dataslice = x[1].drop(labels = ['user_id','item_id'],axis = 1).sort_values('context_timestamp')
    dataslice["user_item_time_diff_1"] = np.insert(np.diff(dataslice["context_timestamp"]).\
             astype(float),0,np.nan)
    dataslice["user_item_time_diff_2"] = np.insert(np.diff(dataslice["context_timestamp"],2).\
             astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),2))
    dataslice["user_item_time_diff_3"] = np.insert(np.diff(dataslice["context_timestamp"],3).\
             astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),3))
    dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True)  
    useritemlists.append(dataslice)
    
usershoplists = Manager().list()  
  
def usershoptimediff(x):
    dataslice = x[1].drop(labels = ['user_id','shop_id'],axis = 1).sort_values('context_timestamp')
    dataslice["user_shop_time_diff_1"] = np.insert(np.diff(dataslice["context_timestamp"]).\
             astype(float),0,np.nan)
    dataslice["user_shop_time_diff_2"] = np.insert(np.diff(dataslice["context_timestamp"],2).\
             astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),2))
    dataslice["user_shop_time_diff_3"] = np.insert(np.diff(dataslice["context_timestamp"],3).\
             astype(float),0,[np.nan]*min(len(dataslice["context_timestamp"]),3))
    dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True) 
    usershoplists.append(dataslice)   


def difftime_parallel(data):    
    useritemgroup = data[['user_id','item_id','instance_id','context_timestamp']].\
    groupby(['user_id','item_id'])
    pool=Pool()
    global useritemlists
    for x in useritemgroup:  
        pool.apply_async(useritemtimediff,args=(x,)) 
    pool.close()
    pool.join()
    feat = pd.concat(useritemlists).reset_index(drop = True)
    del useritemlists
    data = pd.merge(data, feat, on=['instance_id'], how='left') 
    del feat
    
    usershopgroup = data[['user_id','shop_id','instance_id','context_timestamp']].\
    groupby(['user_id','shop_id'])
    pool=Pool()
    global usershoplists
    for x in usershopgroup:  
        pool.apply_async(usershoptimediff,args=(x,)) 
    pool.close()
    pool.join()
    feat = pd.concat(usershoplists).reset_index(drop = True)    
    del usershoplists
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
    data = pd.concat([train, testa,testb])
    del train,testa,testb
    dellabel = data.columns
#%%
    data = difftime_parallel(data) 

    data.drop(labels = dellabel,axis = 1,inplace = True)
    data.to_csv("feature_file_v5.txt",index = False)