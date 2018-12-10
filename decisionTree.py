#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:07:54 2018

@author: Chenjie
"""
import numpy as np
import pandas as pd
from math import log
import operator

training_set = pd.DataFrame.from_csv("preprocessing.csv")
training_set = training_set.as_matrix()

training_data1 = training_set[:,0:2]
training_data2 = training_set[:,3:15]
training_data3 = np.concatenate((training_data1,training_data2),axis = 1)

[row,col] = training_data3.shape

training_data = np.zeros((row,col))

for j in range(col):
    max_val = max(training_data3[:,j])
    for i in range(row):
        training_data[i][j] = training_data3[i][j]/max_val
    
test_set = pd.DataFrame.from_csv("pre_testdata.csv")
test_set = test_set.as_matrix()

test_data1 = test_set[:,0:2]
test_data2 = test_set[:,3:15]
test_data3 = np.concatenate((test_data1,test_data2),axis = 1)
test_label = test_set[:,14]

[row_test,col_test] = test_data3.shape

test_data = np.zeros((row_test,col_test))

for j in range(col_test):
    max_val = max(test_data3[:,j])
    for i in range(row_test):
        test_data[i][j] = test_data3[i][j]/max_val

def calEntropy(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    entropy=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        entropy = entropy - prob*log(prob,2)
    return entropy

def splitDataSet(dataSet,axis,value):
    newdataSet=[]
    for featVector in dataSet:
        if featVector[axis]==value:
            newFeatureVec =featVector[:axis]
            newFeatureVec.extend(featVector[axis+1:])
            newdataSet.append(newFeatureVec)
    return newdataSet

def SelectedFeature(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calEntropy(dataSet)
    bestFeature = 0
    for i in range(numFeatures):
        featList = [featureVal[i] for featureVal in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy = newEntropy + prob*calEntropy(subDataSet)
        if (baseEntropy>newEntropy):
            baseEntropy = newEntropy
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for i in classList:
        if i not in classCount.keys():
            classCount[i]=0
        classCount[i]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createDecisionTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=SelectedFeature(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValue=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValue)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createDecisionTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    currentFeat = list(inputTree.keys())[0]
    secondTree = inputTree[currentFeat]
    featureIndex = featLabels.index(currentFeat)
    classLabel = 0
    try:
        for value in secondTree.keys():
            if value == testVec[featureIndex]:
                if type(secondTree[value]).__name__ == 'dict':
                    classLabel = classify(secondTree[value],featLabels,testVec)
                else:
                    classLabel = secondTree[value]
        return classLabel
    except AttributeError:
        print(secondTree)
         
dataSet = training_data.tolist()
labels=['age','workclass','education','education-num','marital-status','occupation','Transport-moving','relationship','race','sex','capital-gain','capital-loss','hours-per-week']
predict_tree = createDecisionTree(dataSet, labels)
#print(predict_tree)
errorCount = 0
for i in range(row_test):
    new_data = training_data[i,:].tolist()
    labels=['age','workclass','education','education-num','marital-status','occupation','Transport-moving','relationship','race','sex','capital-gain','capital-loss','hours-per-week']
    Label = classify(predict_tree,labels,new_data)
    if Label != test_label[i]:
        errorCount+=1
errorRatio = errorCount/row