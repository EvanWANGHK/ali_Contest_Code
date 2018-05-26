# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:58:18 2018

@author: lzhaoai
"""

#import pandas as pd
#df1 = pd.DataFrame()
#df2 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
#df3 = pd.concat([df1,df2])

#import pandas as pd
import multiprocessing
print(multiprocessing.cpu_count())

#from psutil import virtual_memory
#print(virtual_memory().total*1.0/(1024**3))

#pool = multiprocessing.Pool()
#
#data =  pd.DataFrame()
#def func(i):
#    data = pd.concat([data,pd.DataFrame({'col1': [i]*4})],axis = 1)
#    print(i)
#    return data
#pool.map(func, range(10))


#def func(n):
#    for i in range(10000):
#        for j in range(10000):
#            s=j*i
#    print(n)
#        
#import multiprocessing 
#
#if __name__ == '__main__':
#    pool = multiprocessing.Pool(processes=4)
#    pool.map(func, range(10))
#    pool.close()
#    pool.join()   
#    print('done')


#from itertools import product
#import time
#import numpy as np
#
#pool = multiprocessing.Pool() 
#
#N = 5
#output = np.zeros((N,N))
# #defaults to number of available CPU's
#t = time.time()
#for ind, res in enumerate(pool.imap(func, product(xrange(N), xrange(N)))):
#    print ind
#    output.flat[ind-1] = res   
#pool.close()
#pool.join()  
#print(time.time()-t)      