# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30

@author: Weinan Li

@modified by: Arda C. Bati

This module's purpose is the usage of extract_data function to extract relevant
data from external datasets.

"""

import pandas as pd
import os

def extract_data():
    '''
    Used to extract the gdp & pupulation info from the more complete external 
    datasets. Requires the relevant files to be in the Project_Files directory
    at the top level. Returns the final dataframe which should be the input of 
    the CleanData object.
    
    returns:
        data (DataFrame)
    '''
    
    filename1 = 'Project_Files/Life Expectancy Data.csv'
    filename2 = 'Project_Files/GDP.csv'
    filename3 = 'Project_Files/Population.csv'
    
    assert os.path.isfile(filename1)
    assert os.path.isfile(filename2)
    assert os.path.isfile(filename3)
    
    data=pd.read_csv(filename1)
    gdp=pd.read_csv(filename2,error_bad_lines=False,skiprows=4)
    population=pd.read_csv(filename3,skiprows=4)
    
    features = list(data.columns.values)
    pop_index = features.index('Population')
    GDP_index = features.index('GDP')
    
    pop_new, gdp_new = __fix_names(population.copy(),gdp.copy())
    population, gdp =  pop_new, gdp_new
    
    for i in range(len(data)):
        
        country=data.iloc[i,0]
        year=str(data.iloc[i,1])
        pop=population[population['Country Name']==country][year]
        G_DP=gdp[gdp['Country Name']==country][year]
        #print(float(pop))
        try:
            data.iloc[i,pop_index]=float(pop)
            data.iloc[i,GDP_index]=float(G_DP)
        except:
            #few countries unable to match, show here
            #print(year,country)
            
    return data
        
def __fix_names(population, gdp):
    '''
    Used to correct country naming differences between our data set and the 
    external datasets.
    
    params:
        population: (DataFrame)
        gdp: DataFrame
        
    returns: 
        pop_f: (DataFrame)
        gdp_f: DataFrame
    '''
    
    assert isinstance(population, pd.DataFrame)
    assert isinstance(gdp, pd.DataFrame)
    
    population.loc[population['Country Name'] == 'Kyrgyz Republic','Country Name'] = 'Kyrgyzstan'
    gdp.loc[gdp['Country Name'] == 'Kyrgyz Republic','Country Name'] = 'Kyrgyzstan'
    
    population.loc[population['Country Name'] == 'Congo, Dem. Rep.','Country Name'] = 'Democratic Republic of the Congo'
    gdp.loc[gdp['Country Name'] == 'Congo, Dem. Rep.','Country Name'] = 'Democratic Republic of the Congo'
    
    population.loc[population['Country Name'] == 'Eswatini','Country Name'] = 'Swaziland'
    gdp.loc[gdp['Country Name'] == 'Eswatini','Country Name'] = 'Swaziland'
    
    population.loc[population['Country Name'] == 'C?e d\'Ivoire','Country Name'] = 'Côte d\'Ivoire'
    gdp.loc[gdp['Country Name'] == 'C?e d\'Ivoire','Country Name'] = 'Côte d\'Ivoire'
    
    population.loc[population['Country Name'] == 'Czech Republic','Country Name'] = 'Czechia'
    gdp.loc[gdp['Country Name'] == 'Czech Republic','Country Name'] = 'Czechia'
    
    pop_f = population
    gdp_f = gdp
    
    return pop_f, gdp_f
