import pandas as pd

import warnings
from multiprocessing import Manager,Pool
warnings.filterwarnings("ignore")


userlists = Manager().list()

def usertimecnt(x):
    dataslice = x[1].drop(labels = ['user_id'],axis = 1).sort_values('context_timestamp')       
    user15mincount = func(list(dataslice["context_timestamp"]))       
    dataslice["user_15min_count"] = user15mincount       
    dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True)  
    userlists.append(dataslice)

useritemlists = Manager().list()

def useritemtimecnt(x):
    dataslice = x[1].drop(labels = ['user_id','item_id'],axis = 1).sort_values('context_timestamp')
    useritem15mincount = func(list(dataslice["context_timestamp"]))
    dataslice["user_item_15min_count"] = useritem15mincount
    dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True)    
    useritemlists.append(dataslice)
    
usershoplists = Manager().list()  
  
def usershoptimecnt(x):
    dataslice = x[1].drop(labels = ['user_id','shop_id'],axis = 1).sort_values('context_timestamp')
    usershop15mincount = func(list(dataslice["context_timestamp"]))
    dataslice["user_shop_15min_count"] = usershop15mincount
    dataslice.drop(labels = 'context_timestamp',axis = 1,inplace = True)    
    usershoplists.append(dataslice)   


def func(sorted_list,num = 15*60):
    len_of_list = len(sorted_list)
    output_list = [0] * len_of_list
    ref_list = [i + num for i in sorted_list]
    counter_A = 1
    counter_B = 0
    while counter_A < len_of_list:
       if sorted_list[counter_A] <= ref_list[counter_B]:
          output_list[counter_A] = counter_A - counter_B
          counter_A += 1
       else:
          counter_B += 1
    return output_list 



def timecount_parallel(data):   
    usergroup = data[['user_id','instance_id','context_timestamp']].groupby('user_id')
    pool=Pool()
    global userlists
    for x in usergroup:        
         pool.apply_async(usertimecnt,args=(x,)) 
    pool.close()
    pool.join()         
    feat = pd.concat(userlists).reset_index(drop = True)
    del userlists
    data = pd.merge(data, feat, on=['instance_id'], how='left') 
    del feat    
        
   
    useritemgroup = data[['user_id','item_id','instance_id','context_timestamp']].\
    groupby(['user_id','item_id'])
    pool=Pool()
    global useritemlists
    for x in useritemgroup:                
        pool.apply_async(useritemtimecnt,args=(x,)) 
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
        pool.apply_async(usershoptimecnt,args=(x,)) 
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

    data = timecount_parallel(data) 

    data.drop(labels = dellabel,axis = 1,inplace = True)
    data.to_csv("feature_file_v4.txt",index = False)
