# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:42:16 2020

@author: ZLT
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df1 = pd.read_excel('Person_1801.xls').dropna(axis = 0);
    df2 = pd.read_excel('Person_1802.xls').dropna(axis = 0);
    df=pd.concat([df1,df2])
    
    df=df.drop_duplicates(subset=["身高（cm）","体重（500g）","脚长（cm）"],keep='first')
    df=df.astype('float32')
    #print(df) 

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
    
    model.add(tf.keras.layers.Dense(150,input_shape=(3,),activation='relu'))
    model.add(Dropout(0.1))
    model.add(tf.keras.layers.Dense(20,activation='relu'))
    model.add(Dropout(0.1))
    model.add(tf.keras.layers.Dense(10,activation='relu'))
    model.add(Dropout(0.1))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid')) 
    # model.summary()
     
     
    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['acc'])
     
    history = model.fit(test_x_normal, test_y,epochs=50,validation_split=0.1,batch_size=16,verbose=2)
    
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


   
   