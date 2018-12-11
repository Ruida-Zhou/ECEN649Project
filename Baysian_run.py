#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 00:12:21 2018

@author: Chenjie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep

training_set = pd.DataFrame.from_csv("preprocessing.csv")
training_set = training_set.as_matrix()

[row,col] = training_set.shape

Prob_age = np.zeros((2,col))
Prob_workclass = np.zeros((2,8))
Prob_education = np.zeros((2,16))
Prob_edunum = np.zeros((2,16))
Prob_marriage = np.zeros((2,7))
Prob_occupation = np.zeros((2,14))
Prob_relation = np.zeros((2,6))
Prob_race = np.zeros((2,5))
Prob_sex = np.zeros((2,2))
Prob_capgain = np.zeros((2,20))
Prob_caploss = np.zeros((2,2))
Prob_hours = np.zeros((2,20))
#Prob_hpw = np.zeros()
Priors = np.zeros((2))

for t in range(row):
    if training_set[t,14] == 1:
        Priors[1] = Priors[1] + 1
        if training_set[t,0] >=0 and training_set[t,0] < 10:
            Prob_age[1,0] = Prob_age[1,0] + 1
        elif training_set[t,0] >=10 and training_set[t,0] <20:
            Prob_age[1,1] = Prob_age[1,1] + 1
        elif training_set[t,0] >=20 and training_set[t,0] <30:
            Prob_age[1,2] = Prob_age[1,2] + 1
        elif training_set[t,0] >=30 and training_set[t,0] <40:
            Prob_age[1,3] = Prob_age[1,3] + 1
        elif training_set[t,0] >=40 and training_set[t,0] <50:
            Prob_age[1,4] = Prob_age[1,4] + 1
        elif training_set[t,0] >=50 and training_set[t,0] <60:
            Prob_age[1,5] = Prob_age[1,5] + 1
        elif training_set[t,0] >=60 and training_set [t,0]<70:
            Prob_age[1,6] = Prob_age[1,6] + 1
        elif training_set[t,0] >=70 and training_set[t,0] <80:
            Prob_age[1,7] = Prob_age[1,7] + 1
        elif training_set[t,0] >=80:
            Prob_age[1,8] = Prob_age[1,8] + 1
    elif training_set[t,14] == 0:
        Priors[0] = Priors[0] + 1
        if training_set[t,0] >=0 and training_set[t,0]< 10:
            Prob_age[0,0] = Prob_age[0,0] + 1
        elif training_set[t,0] >=10 and training_set[t,0] <20:
            Prob_age[0,1] = Prob_age[0,1] + 1
        elif training_set[t,0] >=20 and training_set[t,0] <30:
            Prob_age[0,2] = Prob_age[0,2] + 1
        elif training_set[t,0] >=30 and training_set[t,0] <40:
            Prob_age[0,3] = Prob_age[0,3] + 1
        elif training_set[t,0] >=40 and training_set[t,0] <50:
            Prob_age[0,4] = Prob_age[0,4] + 1
        elif training_set[t,0] >=50 and training_set[t,0] <60:
            Prob_age[0,5] = Prob_age[0,5] + 1
        elif training_set[t,0] >=60 and training_set[t,0] <70:
            Prob_age[0,6] = Prob_age[0,6] + 1
        elif training_set[t,0] >=70 and training_set[t,0] <80:
            Prob_age[0,7] = Prob_age[0,7] + 1
        elif training_set[t,0] >=80:
            Prob_age[0,8] = Prob_age[0,8] + 1
            

Prob_age[0,:] = np.divide(Prob_age[0,:],Priors[0])
Prob_age[1,:] = np.divide(Prob_age[1,:],Priors[1])


for k in range(row):
    temp_workclass = training_set[k,1]%8
    temp_education = training_set[k,3]%16
    temp_edunum = training_set[k,4]%16
    temp_marriage = training_set[k,5]%7
    temp_occupation = training_set[k,6]%14
    temp_relation = training_set[k,7]%6
    temp_race = training_set[k,8]%5
    temp_sex = training_set[k,9]%2
    temp_capgain = training_set[k,10]//5000
    temp_caploss = training_set[k,11]//1
    temp_hours = training_set[k,12]//5
    if training_set[k,14] == 1:
        
        Prob_workclass[1,(temp_workclass+7)%8]+= 1

        Prob_education[1,(temp_education+15)%16]+= 1
        
        Prob_edunum[1,(temp_edunum+15)%16]+= 1
            
        Prob_marriage[1,(temp_marriage+6)%7]+= 1
            
        Prob_occupation[1,(temp_occupation+13)%14]+= 1
            
        Prob_relation[1,(temp_relation+5)%6]+= 1

        Prob_race[1,(temp_race+4)%5]+= 1
        
        if temp_sex == 1:
            Prob_sex[1,0]+= 1
        elif temp_sex == 0:
            Prob_sex[1,1]+= 1
            
        Prob_capgain[1,temp_capgain]+= 1
            
        if temp_caploss == 0:
            Prob_caploss[1,0]+= 1
        elif temp_caploss != 0:
            Prob_caploss[1,1]+= 1
            
        Prob_hours[1,temp_hours]+= 1
        
            
    elif training_set[k,14] == 0:
        Prob_workclass[0,(temp_workclass+7)%8]+= 1

        Prob_education[0,(temp_education+15)%16]+= 1
        
        Prob_edunum[0,(temp_edunum+15)%16]+= 1
            
        Prob_marriage[0,(temp_marriage+6)%7]+= 1
            
        Prob_occupation[0,(temp_occupation+13)%14]+= 1
            
        Prob_relation[0,(temp_relation+5)%6]+= 1

        Prob_race[0,(temp_race+4)%5]+= 1

        if temp_sex == 1:
            Prob_sex[0,0]+= 1
        elif temp_sex == 0:
            Prob_sex[0,1]+= 1
            
        Prob_capgain[0,temp_capgain]+= 1
            
        if temp_caploss == 0:
            Prob_caploss[0,0]+= 1
        elif temp_caploss != 0:
            Prob_caploss[0,1]+= 1
            
        Prob_hours[0,temp_hours]+= 1

Prob_workclass[0,:] = np.divide(Prob_workclass[0,:],Priors[0])
Prob_workclass[1,:] = np.divide(Prob_workclass[1,:],Priors[1])

Prob_education[0,:] = np.divide(Prob_education[0,:],Priors[0])
Prob_education[1,:] = np.divide(Prob_education[1,:],Priors[1])

Prob_edunum[0,:] = np.divide(Prob_edunum[0,:],Priors[0])
Prob_edunum[1,:] = np.divide(Prob_edunum[1,:],Priors[1])

Prob_marriage[0,:] = np.divide(Prob_marriage[0,:],Priors[0])
Prob_marriage[1,:] = np.divide(Prob_marriage[1,:],Priors[1])

Prob_occupation[0,:] = np.divide(Prob_occupation[0,:],Priors[0])
Prob_occupation[1,:] = np.divide(Prob_occupation[1,:],Priors[1])

Prob_relation[0,:] = np.divide(Prob_relation[0,:],Priors[0])
Prob_relation[1,:] = np.divide(Prob_relation[1,:],Priors[1])

Prob_race[0,:] = np.divide(Prob_race[0,:],Priors[0])
Prob_race[1,:] = np.divide(Prob_race[1,:],Priors[1])

Prob_sex[0,:] = np.divide(Prob_sex[0,:],Priors[0])
Prob_sex[1,:] = np.divide(Prob_sex[1,:],Priors[1])

Prob_capgain[0,:] = np.divide(Prob_capgain[0,:],Priors[0])
Prob_capgain[1,:] = np.divide(Prob_capgain[1,:],Priors[1])

Prob_caploss[0,:] = np.divide(Prob_caploss[0,:],Priors[0])
Prob_caploss[1,:] = np.divide(Prob_caploss[1,:],Priors[1])

Prob_hours[0,:] = np.divide(Prob_hours[0,:],Priors[0])
Prob_hours[1,:] = np.divide(Prob_hours[1,:],Priors[1])

Priors[0] = float(Priors[0])/row
Priors[1] = float(Priors[1])/row

test_set = pd.DataFrame.from_csv("pre_testdata.csv")
test_set = test_set.as_matrix()

[r,c] =  test_set.shape

prediction = np.zeros((r,1))

for p in range(r):
    Cond_probability0 = 1
    Cond_probability1 = 1
    Cond_probability0 *= Prob_age[0,test_set[p,0]//10]
    Cond_probability0 *= Prob_workclass[0,(test_set[p,1]+7)%8]
    Cond_probability0 *= Prob_education[0,(test_set[p,3]+15)%16]
    Cond_probability0 *= Prob_edunum[0,(test_set[p,4]+15)%16]   
    Cond_probability0 *= Prob_marriage[0,(test_set[p,5]+6)%7]
    Cond_probability0 *= Prob_occupation[0,(test_set[p,6]+13)%14]
    Cond_probability0 *= Prob_relation[0,(test_set[p,7]+5)%6]
    Cond_probability0 *= Prob_race[0,(test_set[p,8]+4)%5]
    Cond_probability0 *= Prob_sex[0,(test_set[p,9]+1)%2]
    Cond_probability0 *= Prob_capgain[0,test_set[p,10]//5000]
    if test_set[p,10]//1 != 0:
        temp = 1
    else:
        temp = 0
    Cond_probability0 *= Prob_caploss[0,temp]
    Cond_probability0 *= Prob_hours[0,test_set[p,12]//5]
    Cond_probability0 *= Priors[0]
    
    Cond_probability1 *= Prob_age[1,test_set[p,0]//10]
    Cond_probability1 *= Prob_workclass[1,(test_set[p,1]+7)%8]
    Cond_probability1 *= Prob_education[1,(test_set[p,3]+15)%16]
    Cond_probability1 *= Prob_edunum[1,(test_set[p,4]+15)%16]   
    Cond_probability1 *= Prob_marriage[1,(test_set[p,5]+6)%7]
    Cond_probability1 *= Prob_occupation[1,(test_set[p,6]+13)%14]
    Cond_probability1 *= Prob_relation[1,(test_set[p,7]+5)%6]
    Cond_probability1 *= Prob_race[1,(test_set[p,8]+4)%5]
    Cond_probability1 *= Prob_sex[1,(test_set[p,9]+1)%2]
    Cond_probability1 *= Prob_capgain[1,test_set[p,10]//5000]
    if test_set[p,10]//1 != 0:
        temp = 1
    else:
        temp = 0
    Cond_probability1 *= Prob_caploss[1,temp]
    Cond_probability1 *= Prob_hours[1,test_set[p,12]//5]
    Cond_probability1 *= Priors[1]
    
    if Cond_probability0 >= Cond_probability1:
        prediction[p] = 0
    else:
        prediction[p] = 1
    
error_ratio = 0
for i in range(r):
    if test_set[i,14]!= prediction[i]:
        error_ratio = error_ratio + 1
error_ratio = error_ratio/r
Testdata = np.hstack((test_set,prediction))
dftest = pd.DataFrame(Testdata)

dftest.to_csv("Baysian_for_test_data.csv")    









