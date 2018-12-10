#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:10:02 2018

@author: Chenjie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep

training_set = pd.DataFrame.from_csv("adult.csv",index_col = None, header = None)
training_set = training_set.as_matrix()

[training_row, training_col] = training_set.shape    

for t in range(training_row):
    if training_set[t,14] == ' <=50K.':
        training_set[t,14] = 0
    elif training_set[t,14] == ' >50K.':
        training_set[t,14] = 1
        
    if training_set[t,1] == ' Private':
        training_set[t,1] = 1
    elif training_set[t,1] == ' Self-emp-not-inc':
        training_set[t,1] = 2
    elif training_set[t,1] == ' Self-emp-inc':
        training_set[t,1] = 3
    elif training_set[t,1] == ' Federal-gov':
        training_set[t,1] = 4
    elif training_set[t,1] == ' Local-gov':
        training_set[t,1] = 5
    elif training_set[t,1] == ' State-gov':
        training_set[t,1] = 6
    elif training_set[t,1] == ' Without-pay':
        training_set[t,1] = 7
    elif training_set[t,1] == ' Never-worked':
        training_set[t,1] = 8
        
    if training_set[t,3] == ' Bachelors':
        training_set[t,3] = 1
    elif training_set[t,3] == ' Some-college':
        training_set[t,3] = 2
    elif training_set[t,3] == ' 11th':
        training_set[t,3] = 3
    elif training_set[t,3] == ' HS-grad':
        training_set[t,3] = 4
    elif training_set[t,3] == ' Prof-school':
        training_set[t,3] = 5
    elif training_set[t,3] == ' Assoc-acdm':
        training_set[t,3] = 6
    elif training_set[t,3] == ' Assoc-voc':
        training_set[t,3] = 7
    elif training_set[t,3] == ' 9th':
        training_set[t,3] = 8
    elif training_set[t,3] == ' 7th-8th':
        training_set[t,3] = 9
    elif training_set[t,3] == ' 12th':
        training_set[t,3] = 10
    elif training_set[t,3] == ' Masters':
        training_set[t,3] = 11
    elif training_set[t,3] == ' 1st-4th':
        training_set[t,3] = 12
    if training_set[t,3] == ' 10th':
        training_set[t,3] = 13
    elif training_set[t,3] == ' Doctorate':
        training_set[t,3] = 14
    elif training_set[t,3] == ' 5th-6th':
        training_set[t,3] = 15
    elif training_set[t,3] == ' Preschool':
        training_set[t,3] = 16
        
    if training_set[t,5] == ' Married-civ-spouse':
        training_set[t,5] = 1
    elif training_set[t,5] == ' Divorced':
        training_set[t,5] = 2
    elif training_set[t,5] == ' Never-married':
        training_set[t,5] = 3
    elif training_set[t,5] == ' Separated':
        training_set[t,5] = 4
    elif training_set[t,5] == ' Widowed':
        training_set[t,5] = 5
    elif training_set[t,5] == ' Married-spouse-absent':
        training_set[t,5] = 6
    elif training_set[t,5] == ' Married-AF-spouse':
        training_set[t,5] = 7
    
    if training_set[t,6] == ' Tech-support':
        training_set[t,6] = 1
    elif training_set[t,6] == ' Craft-repair':
        training_set[t,6] = 2
    elif training_set[t,6] == ' Other-service':
        training_set[t,6] = 3
    elif training_set[t,6] == ' Sales':
        training_set[t,6] = 4
    elif training_set[t,6] == ' Exec-managerial':
        training_set[t,6] = 5
    elif training_set[t,6] == ' Prof-specialty':
        training_set[t,6] = 6
    elif training_set[t,6] == ' Handlers-cleaners':
        training_set[t,6] = 7
    elif training_set[t,6] == ' Machine-op-inspct':
        training_set[t,6] = 8
    elif training_set[t,6] == ' Adm-clerical':
        training_set[t,6] = 9
    elif training_set[t,6] == ' Farming-fishing':
        training_set[t,6] = 10
    elif training_set[t,6] == ' Transport-moving':
        training_set[t,6] = 11
    elif training_set[t,6] == ' Priv-house-serv':
        training_set[t,6] = 12
    if training_set[t,6] == ' Protective-serv':
        training_set[t,6] = 13
    elif training_set[t,6] == ' Armed-Forces':
        training_set[t,6] = 14
        
    if training_set[t,7] == ' Wife':
        training_set[t,7] = 1
    elif training_set[t,7] == ' Own-child':
        training_set[t,7] = 2
    elif training_set[t,7] == ' Husband':
        training_set[t,7] = 3
    elif training_set[t,7] == ' Not-in-family':
        training_set[t,7] = 4
    elif training_set[t,7] == ' Other-relative':
        training_set[t,7] = 5
    elif training_set[t,7] == ' Unmarried':
        training_set[t,7] = 6
        
    if training_set[t,8] == ' White':
        training_set[t,8] = 1
    elif training_set[t,8] == ' Asian-Pac-Islander':
        training_set[t,8] = 2
    elif training_set[t,8] == ' Amer-Indian-Eskimo':
        training_set[t,8] = 3
    elif training_set[t,8] == ' Other':
        training_set[t,8] = 4
    elif training_set[t,8] == ' Black':
        training_set[t,8] = 5
        
    if training_set[t,9] == ' Female':
        training_set[t,9] = 1
    elif training_set[t,9] == ' Male':
        training_set[t,9] = 2
        
    if training_set[t,13] == ' United-States':
        training_set[t,13] = 1
    elif training_set[t,13] == ' Cambodia':
        training_set[t,13] = 2
    elif training_set[t,13] == ' England':
        training_set[t,13] = 3
    elif training_set[t,13] == ' Puerto-Rico':
        training_set[t,13] = 4
    elif training_set[t,13] == ' Canada':
        training_set[t,13] = 5
    elif training_set[t,13] == ' Germany':
        training_set[t,13] = 6
    elif training_set[t,13] == ' Outlying-US(Guam-USVI-etc)':
        training_set[t,13] = 7
    elif training_set[t,13] == ' India':
        training_set[t,13] = 8
    elif training_set[t,13] == ' Japan':
        training_set[t,13] = 9
    elif training_set[t,13] == ' Greece':
        training_set[t,13] = 10
    elif training_set[t,13] == ' South':
        training_set[t,13] = 11
    elif training_set[t,13] == ' China':
        training_set[t,13] = 12
    if training_set[t,13] == ' Cuba':
        training_set[t,13] = 13
    elif training_set[t,13] == ' Iran':
        training_set[t,13] = 14
    elif training_set[t,13] == ' Honduras':
        training_set[t,13] = 15
    elif training_set[t,13] == ' Philippines':
        training_set[t,13] = 16
    if training_set[t,13] == ' Italy':
        training_set[t,13] = 17
    elif training_set[t,13] == ' Poland':
        training_set[t,13] = 18
    elif training_set[t,13] == ' Jamaica':
        training_set[t,13] = 19
    elif training_set[t,13] == ' Vietnam':
        training_set[t,13] = 20
    elif training_set[t,13] == ' Mexico':
        training_set[t,13] = 21
    elif training_set[t,13] == ' Portugal':
        training_set[t,13] = 22
    if training_set[t,13] == ' Ireland':
        training_set[t,13] = 23
    elif training_set[t,13] == ' France':
        training_set[t,13] = 24
    elif training_set[t,13] == ' Dominican-Republic':
        training_set[t,13] = 25
    elif training_set[t,13] == ' Laos':
        training_set[t,13] = 26
    if training_set[t,13] == ' Ecuador':
        training_set[t,13] = 27
    elif training_set[t,13] == ' Taiwan':
        training_set[t,13] = 28
    elif training_set[t,13] == ' Haiti':
        training_set[t,13] = 29
    elif training_set[t,13] == ' Columbia':
        training_set[t,13] = 30
    elif training_set[t,13] == ' Hungary':
        training_set[t,13] = 31
    elif training_set[t,13] == ' Guatemala':
        training_set[t,13] = 32
    if training_set[t,13] == ' Nicaragua':
        training_set[t,13] = 33
    elif training_set[t,13] == ' Scotland':
        training_set[t,13] = 34
    elif training_set[t,13] == ' Thailand':
        training_set[t,13] = 35
    elif training_set[t,13] == ' Yugoslavia':
        training_set[t,13] = 36
    if training_set[t,13] == ' El-Salvador':
        training_set[t,13] = 37
    elif training_set[t,13] == ' Trinadad&Tobago':
        training_set[t,13] = 38
    elif training_set[t,13] == ' Peru':
        training_set[t,13] = 39
    elif training_set[t,13] == ' Hong':
        training_set[t,13] = 40
    elif training_set[t,13] == ' Holand-Netherlands':
        training_set[t,13] = 41
    

        
data_set = np.zeros((1,training_col))

for a in range(training_row):
    for b in range(training_col):
        if training_set[a,b] == ' ?':
            break
        if b == training_col-1:
           data_set = np.row_stack((data_set,training_set[a,:]))

data_set = np.delete(data_set,(0), axis=0)
dftest = pd.DataFrame(data_set)
dftest.to_csv("pre_testdata.csv")