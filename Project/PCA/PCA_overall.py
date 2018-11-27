#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:20:43 2018
@author: jinqingyuan
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Project.Data_Cleaning import clean_data

class PCA_Analysis:
    '''
    This class is used to determine the most relevant features of the dataset, 
    through Principal Component Analysis (PCA).
    '''

    def __init__(self,data):
        '''
        Runs the main PCA algorythm on the cleaned data.
        
        param: data object from class CleanData.
        
        variables:
        x(datatype:np.ndarray)
        eig_vals(datatype:np.ndarray)
        finalDf(datatype:np.ndarray)
        features(datatype:list)
        '''
        self.__x = 0
        self.__eig_vals = 0
        self.__finalDf = 0
        self.__features = 0
        
        assert isinstance(data, clean_data.CleanData)
        
        df = data.modified
        n = 1000
        x = df.head(n)
        
        '''Separating out the features'''
        features = ['Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure','Hepatitis B','Measles','BMI','under-five deaths','Polio','Total expenditure','Diphtheria',' HIV/AIDS','GDP','Population','thinness  1-19 years','thinness 5-9 years','Income composition of resources','Schooling']
        x = df.ix[:,4:23].values
        x = np.array(x,dtype = float)
        x = np.round(x)
        x = np.array(x,dtype = int)
        
        ''' Separating out the target'''
        y = df.loc[:,['Country','Year','Status','Life expectancy ']].values
        
        '''Standardizing the features'''
        x_std = StandardScaler().fit_transform(x)
        
        '''Reducing the datasize'''
        pca = PCA(n_components=5)
        principalComponents = pca.fit_transform(x_std)
        principalDf = pd.DataFrame(data = principalComponents , columns = ['pc1', 'pc2','pc3','pc4','pc5'])
        finalDf = pd.concat([principalDf, df[['Country','Year','Status','Life expectancy ']]], axis = 1)
        
        if finalDf.shape[0] > df.shape[0]:
            diff = finalDf.shape[0] - df.shape[0];
            finalDf = finalDf[:-diff]
        
        '''Correlation Matrix'''
        ''' eig_vals, eig_vecs are what we should care about'''
        
        mean_vec = np.mean(x_std, axis=0)
        cov_mat = np.array((x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0]-1))
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs)
        
        self.__x = x
        self.__eig_vals = eig_vals
        self.__finalDf = finalDf
        self.__features = features
        
        #print('Eigenvectors \n%s' %eig_vecs)
        #print('\nEigenvalues \n%s' %eig_vals)
        
    @property
    def x(self):
        return self.__x.copy()
    	
    @property
    def eig_vals(self):
        return self.__eig_vals.copy()
    
    @property
    def finalDf(self):
        return self.__finalDf.copy()
    
    @property
    def features(self):
        return self.__features.copy()


def calnum(eig_vals):
    '''
    Return when new subspace could provide enough information(over 85%)
    Input:
    eig_vals(datatype:list)
    Output:
    Nmin(datatype:int)
    '''
    assert isinstance(eig_vals,np.ndarray)
    
    Len = len(eig_vals)
    Pe = 0
    Nmin = 0
    for i in range(Len):
        Pe = Pe + eig_vals[i]
        Ppe = Pe/sum(eig_vals)
        if Ppe > 0.85:
            Nmin = i+1
            break
    return Nmin


def calcoefficient(x,finalDf,features):
    '''
    Analyse each coefficient of variable in sample data
    a1,a2,a3,a4,a5 are different coefficient combinations regarding relevance
    Return 4 negative components and 4 positive components.
    
    Inputs:
    x(datatype:np.ndarray)
    finalDf(datatype:np.ndarray)
    features(datatype:list)
    
    Outputs:
    pcoef(datatype:list): implies the index of each principle component in list'features'
    finalcomp(datatype:list):shows the most four negative components and most four positive components
    '''
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
