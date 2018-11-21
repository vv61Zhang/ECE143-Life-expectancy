#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:49:51 2018

@author: jinqingyuan
"""

def calnum(eig_vals):
    '''
    Determine when new subspace could provide enough relevant information(over 85%)
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
