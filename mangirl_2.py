# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:42:16 2020

@author: ZLT
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    df1 = pd.read_excel('Person_1801.xls').dropna(axis = 0);
    df2 = pd.read_excel('Person_1802.xls').dropna(axis = 0);
    df=pd.concat([df1,df2])
    df=df.drop_duplicates(subset=["身高（cm）","体重（500g）","脚长（cm）"],keep='first')
    df=df.astype('float32')
    #print(df) 

    #---------------------------------------
    
    k = 2 
    threshold = 2 
    iteration = 500 
    
    data=df
    data = data.iloc[:, 0:-1]
    data_zs = 1.0*(data - data.mean())/data.std()
    
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) 
    model.fit(data_zs) 
    
   
    r = pd.concat([data_zs, pd.Series(model.labels_, index = data.index)], axis = 1)  
    r.columns = list(data.columns) + [u'聚类类别'] 
    
    norm = []
    for i in range(k):
        norm_tmp = r[["身高（cm）", "体重（500g）", "脚长（cm）"]][r[u'聚类类别'] == i]-model.cluster_centers_[i]  
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis = 1) 
        norm.append(norm_tmp/norm_tmp.median()) 
    
    norm = pd.concat(norm) 
    
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    norm[norm <= threshold].plot(style = 'go') 
    
    discrete_points = norm[norm > threshold] 
    discrete_points.plot(style = 'ro')
    
    plt.xlabel(u'编号')
    plt.ylabel(u'相对距离')
    plt.show()
    
    #print(norm)
    #norm.to_csv('norm.csv',header=False,index=False,encoding='UTF-8')
    
    index_=0
    
    df=df.reset_index(drop=True)
    data=df
    for i in norm:
        if i>threshold:
            print(data.iloc[index_])
            df=df.drop(index_)
        index_+=1
    #---------------------------------------
    print(df)
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    nor_df = (df-df.min())/(df.max()-df.min())
    nor_df.to_csv('normalize.csv',header=False,index=False,encoding='UTF-8')
    
    x_normal = nor_df.iloc[:, 0:-1]
    y_normal = nor_df.iloc[:, -1]
    #print(x,y)
 
    df_test=pd.read_excel('Person_test.xls').dropna(axis = 0);
    df_test=df_test.astype('float32')
    test_x = df_test.iloc[:, 0:-1]
    
    
    nor_testdf = (df_test-df_test.min())/(df_test.max()-df_test.min())
    nor_testdf.to_csv('text_normalize.csv',header=False,index=False,encoding='UTF-8')
    
    test_x_normal = nor_testdf.iloc[:, 0:-1]
    test_y = nor_testdf.iloc[:, -1]
    
    from tensorflow.keras.layers import Dropout
    
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(137,input_shape=(3,),activation='relu'))
    model.add(Dropout(0.1))
    model.add(tf.keras.layers.Dense(30,activation='relu'))
    model.add(Dropout(0.1))
    model.add(tf.keras.layers.Dense(10,activation='relu'))
    model.add(Dropout(0.1))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid')) 
    # model.summary()
     
     
    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['acc'])
     
    history = model.fit(x_normal, y,epochs=150,validation_split=0.1,batch_size=32,verbose=2)
    
    
    scores=model.evaluate(test_x_normal, test_y)
    print('accuracy=',scores)
    
    # history.history.key()  # ['loss', 'acc']
       
    plt.plot(history.epoch, history.history.get('loss'),'bo',label='loss')
    plt.plot(history.epoch, history.history.get('acc'),'g',label='acc')
    plt.xlabel('Epochs')
    plt.ylabel('loss+acc')
    plt.legend(loc='best')  
    plt.show()
    

    print(test_x_normal)
    print(x_normal)


   
   