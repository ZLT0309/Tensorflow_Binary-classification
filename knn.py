# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:20:21 2020

@author: ZLT
"""

import numpy as np
from xlrd import open_workbook
import operator

def load_file(filename):
        workbook = open_workbook(filename)
        sheet = workbook.sheet_by_index(0)
        height = sheet.col_values(0)
        weight = sheet.col_values(1)
        length = sheet.col_values(2)
        label = sheet.col_values(3)
        height.pop(0)
        weight.pop(0)
        length.pop(0)
        label.pop(0)
        
        data = list(zip(list(height),list(weight),list(length)))
        data = [list(i) for i in data]
        data = np.array(data).reshape(len(height),3)
        
        label = list(map(int, label))
        
        return data, label
    
def normalize(data):
    n_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return n_data

def Judge(inX, data, label, k):
    dataSetsize = len(data)
    diffMat = np.tile(inX, (dataSetsize,1)) - data
    sqDiffMat = diffMat **2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    
    for i in range(k):
        voteIlabel = label[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    filename = 'Person_1802.xlsx'
    outputfile = 'result.txt'
    
    data, label = load_file(filename)
    #print(data)
    #print(label)
    
    #data = normalize(data)
    #print(data)
    
    testdata = data
    testlabel = label
    
    err = []
    err_data = {}
    
    for i in range(10):
        count = 0 
        for k in range(len(testdata)):
            rtn = Judge(testdata[k],data,label, 5)
            if rtn != testlabel[k]:
                count+=1;
                err_data[k] = err_data.get(k,0) + 1
        err.append(count)
    
    with open(outputfile, 'a+') as fp:
        fp.write('每次的错误次数分别为：' + str(err) + '\n')
        err = [i/len(testdata) for i in err]
        avg_err = "%.4f%%" % (sum(err)/10 *100)
        fp.write('平均错误率为:'+str(avg_err)+'\n')
        for key,value in err_data.items():
            fp.write('第'+str(key)+'个测试样本'+str(testdata[key]) + 
                     '标签为'+str(testlabel[key])+'的识别错误次数为：'+ str(value)+'\n')
