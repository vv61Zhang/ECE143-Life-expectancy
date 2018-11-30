#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:51:29 2018

@author: Qingyuan Jin
"""


def calcoefficient(x,finalDf,features):
    '''
    Analyse each coefficient of variable in sample data
    a1,a2,a3,a4,a5 are different coefficient combinations regarding relevance
    Return 4 negative components and 4 positive components
    Inputs:
    x(datatype:np.ndarray)
    finalDf(datatype:np.ndarray)
    features(datatype:list)
    Outputs:
    pcoef(datatype:list): implies the index of each principle component in list'features'
    finalcomp(datatype:list):shows the most four negative components and most four positive components
    '''jinqingyuan
    assert isinstance(x,np.ndarray)
    assert isinstance(finalDf,pd.core.frame.DataFrame)
    assert isinstance(features,list)
    
    a1 = np.invert(x) @ finalDf.pc1
    a2 = np.invert(x) @ finalDf.pc2 
    a3 = np.invert(x) @ finalDf.pc3
    a4 = np.invert(x) @ finalDf.pc4
    a5 = np.invert(x) @ finalDf.pc5
    L = np.array([a1,a2,a3,a4,a5])
    A = np.array([sorted(a1),sorted(a2),sorted(a3),sorted(a4),sorted(a5)])
    finala = []
    for i in range(5):
        for j in [0,1,2,3,14,15,16,17]:
            for n in range(18):
                if A[i,j]  == L[i,n]:
                    finala.append(n)
                
    pcoef = [finala[0:8],finala[8:16],finala[16:24],finala[24:32],finala[32:40]]
    comp = []
    for i in range(5):
        for j in pcoef[i]:
            comp.append(features[j])
    finalcomp = [comp[0:8],comp[8:16],comp[16:24],comp[24:32],comp[32:40]]
    return pcoef,finalcomp
