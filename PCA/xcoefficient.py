#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:51:29 2018

@author: jinqingyuan
"""


def calcoefficient(x,finalDf):
    '''
    analyse each coefficient of variable in sample data
    a1,a2,a3,a4,a5 are different coefficient combinations regarding relevance
    '''
    assert isinstance(x,np.ndarray)
    assert isinstance(finalDf,pd.core.frame.DataFrame)
    L = []
    for i in range(5):
    a1 = np.invert(x) @ finalDf.pc1
    a2 = np.invert(x) @ finalDf.pc2 
    a3 = np.invert(x) @ finalDf.pc3
    a4 = np.invert(x) @ finalDf.pc4
    a5 = np.invert(x) @ finalDf.pc5
    return a1,a2,a3,a4,a5
