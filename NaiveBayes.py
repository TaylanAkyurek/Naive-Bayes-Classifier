#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:52:23 2022

@author: alitaylanakyurek
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw01_data_points.csv",dtype = str ,delimiter = ",")
y = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw01_class_labels.csv", delimiter = ",")

x_train = x[:300]
x_test = x[300:]

y_train = y[:300]
y_test = y[300:]


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


N_train = x_train.shape[0]
N_test = x_test.shape[0]

nucleocidCount = 7
classCount = 2

print("x train:")
print(x_train)
print("\n")
print("x test:")
print(x_test)
print("\n")

print("y train:")
print(y_train)
print("\n")
print("y test:")
print(y_test)

pAcd = [[0.0]*nucleocidCount]*classCount
pCcd = [[0.0]*nucleocidCount]*classCount
pGcd = [[0.0]*nucleocidCount]*classCount
pTcd = [[0.0]*nucleocidCount]*classCount

pAcd = np.array(pAcd)
pCcd = np.array(pCcd)
pGcd = np.array(pGcd)
pTcd = np.array(pTcd)

class_priors = [np.mean(y_train == (c + 1)) for c in range(classCount)]


def oneLocationProbability(nucleocid,location,classNum):
    total = 0
    for i in range(N_train):
        if(x_train[i][location] == nucleocid and y_train[i] == classNum):
            total += 1
    #y_train[i] == classNum is actually 1 function for y
    #and x_train[i][location] == nucleocid is 1 function for x
    probability = total/(N_train * class_priors[classNum - 1])
    #also statement above is an extension of 1 function for y(also maps the total to probability)
    return probability
    
for i in range(classCount):
    for j in range(nucleocidCount):
        pAcd[i][j] = oneLocationProbability('A', j, i+1)    
        
        
for i in range(classCount):
    for j in range(nucleocidCount):
        pCcd[i][j] = oneLocationProbability('C', j, i+1)       
            
for i in range(classCount):
    for j in range(nucleocidCount):
        pGcd[i][j] = oneLocationProbability('G', j, i+1)       
            
for i in range(classCount):
    for j in range(nucleocidCount):
        pTcd[i][j] = oneLocationProbability('T', j, i+1)       
            
print("pAcd:")
print(pAcd)
print("\n")
print("pCcd:")
print(pCcd)
print("\n")
print("pGcd:")
print(pGcd)
print("\n")
print("pTcd:")
print(pTcd)
print("\n")
print("class priors:")
print(class_priors)
print("\n")

def g(c,x):
    
    totalScore = 0.0
    for i in range(nucleocidCount):
        if(x[i] == 'A'):
            totalScore += math.log(pAcd[c-1][i])
        elif(x[i] == 'C'):
            totalScore += math.log(pCcd[c-1][i])
        elif(x[i] == 'G'):
            totalScore += math.log(pGcd[c-1][i])
        elif(x[i] == 'T'):
            totalScore += math.log(pTcd[c-1][i])
    
    totalScore += math.log(class_priors[c-1]) 
    return totalScore
    
        
def calculateMatrix(x,y):
    N = x.shape[0]
   
    matrix = [[0]*2]*2
    matrix = np.array(matrix)
    y_truth = 0
    y_pred = 0
    
    for i in range(N):
        if((g(1,x[i]) > g(2,x[i])) and y[i] == 1):
            matrix[0][0] += 1
        elif((g(2,x[i]) > g(1,x[i])) and y[i] == 1):
            matrix[1][0] += 1
        elif((g(1,x[i]) > g(2,x[i])) and y[i] == 2):
            matrix[0][1] += 1
        elif((g(2,x[i]) > g(1,x[i])) and y[i] == 2):
            matrix[1][1] += 1

    return matrix

print("train set confusion matrix:")
print(calculateMatrix(x_train, y_train))
print("\n")
print("test set confusion matrix:")
print(calculateMatrix(x_test, y_test))

            
    