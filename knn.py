#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:22:11 2018

@author: Chenjie
"""
import numpy as np
import pandas as pd
        
training_set = pd.DataFrame.from_csv("preprocessing.csv")
training_set = training_set.as_matrix()

training_data1 = training_set[:,0:2]
training_data2 = training_set[:,3:12]
training_data3 = np.concatenate((training_data1,training_data2),axis = 1)
training_label = training_set[:,14]

[row,col] = training_data3.shape

training_data = np.zeros((row,col))

for j in range(col):
    max_val = max(training_data3[:,j])
    for i in range(row):
        training_data[i][j] = training_data3[i][j]/max_val

test_set = pd.DataFrame.from_csv("pre_testdata.csv")
test_set = test_set.as_matrix()

test_data1 = test_set[:,0:2]
test_data2 = test_set[:,3:12]
test_data3 = np.concatenate((test_data1,test_data2),axis = 1)
test_label = test_set[:,14]

[row_test,col_test] = test_data3.shape

test_data = np.zeros((row_test,col_test))

for j in range(col_test):
    max_val = max(test_data3[:,j])
    for i in range(row_test):
        test_data[i][j] = test_data3[i][j]/max_val

def getDistance(point,new_data):
    [r,c] = new_data.shape
    reference = np.zeros((r,c))
    reference[:] = point                
    minusSquare = (new_data - reference)**2  
    Distance = np.sqrt(minusSquare.sum(axis=1))
    return Distance

def getClass(point,labels,k):
    distance = getDistance(point,training_data)
    argsort = distance.argsort(axis=0)
    classList = list(labels[argsort[0:k]])
    classCount = {}
    for i in classList:
        if i not in classCount:
            classCount[i] = 1
        else:
            classCount[i] += 1
    maxCount = 0
    predict_label = 0
    for key in classCount.keys():
        if classCount[key] > maxCount:
            maxCount = classCount[key]
            predict_label = key
    return predict_label
        
def knn(testData,k):
    labels = []
    for i in range(len(testData)):
        point = testData[i,:]
        label = getClass(point,training_label,k)
        labels.append(label)
    return labels

def errorRatio(knn_class,real_class):
    error = 0
    allCount = len(real_class)
    real_class = list(real_class)
    for i in range(allCount):
        if knn_class[i] != real_class[i]:
            error += 1
    return error/allCount

predict_class = knn(test_data,4)
error_ratio = errorRatio(predict_class,test_label)