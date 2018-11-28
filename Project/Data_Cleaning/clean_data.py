"""
Created on Mon Nov 12 2018
@author: Arda C. Bati
Does data cleaning on the WHO Life expectancy data. Generates different cleaned 
versions of the data in the directory ./output. The calss should run in the same
folder/directory with the file 'Life Expectancy Data.csv', which is provided on 
Group 5's github page. 
The create_csvs() function will create .csv files from the variables obtained in 
the code.
"""

import pandas as pd
import numpy as np
import logging, sys
import os

class CleanData:
    '''
    Class used to clean the dataset 'Life Expectancy Data.csv'
    '''
    
    def __init__(self,log_level=logging.INFO):
        '''
        Does data cleaning on the dataset obtained from 'Life Expectancy Data.csv'
        which should be in the same folder with where this function is ran. Contains
        the variables given below:
        
        NoNaN: 
            The version of the dataset with rows containing any NaN value removed.
        NaN: 
            The version of the dataset with 0 values replaced with NaN
        modified: 
            The version of the dataset where NaN values are filled with interpolation / means.
        feature_tables: 
            list containing seperate dataframes for each frame from the data.
        indices: 
            The indices of the above mentioned feature_tables list.
        
        param: log_level
            chooses the logging level of the initializer
        '''
        
        self.__NoNaN = 0
        self.__NaN = 0
        self.__modified = 0
        self.__feature_tables = 0
        self.indices = 0
        
        assert isinstance(log_level,int)
        assert 0 <= log_level <= 50
        
        # Logging startup
        logging.basicConfig(stream=sys.stdout, level=log_level)
        #format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Analyzing the data
        filename = './Life Expectancy Data.csv'
        assert os.path.isfile(filename),'Please copy "Life Expectancy Data.csv" to the current directory.'
            
        df = pd.read_csv('Life Expectancy Data.csv')
               
        df.loc[df['Status'] == 'Developing','Status'] = 1
        df.loc[df['Status'] == 'Developed','Status'] = 2
        
        columns = list(df.columns.values)
        
        countries_years = df.groupby('Country').count()['Year']
        countries_missing_years = countries_years[countries_years != 16]
        
        # Removing countries that have only one year entry from the dataset
        for i in countries_missing_years.index:
            df = df[df.Country != i]
        
        # Replacing 0 values with NaN
        NaN = df.copy()
        NaN = df.replace(0,np.nan)
        
        # Dropping rows containing NaN in any of their columns
        NoNaN = NaN.copy()
        NoNaN = NoNaN.dropna(0)
         
        # Going over each feature in the dataset, dropping rows containing NaN
        indices = [columns[2]] + columns[4:]
        feature_tables = [df[['Country','Year','Life expectancy ', i]].dropna(0)  for i in indices]
        feature_tables.append(df[['Country','Year','Life expectancy ']].dropna(0))
        
        logging.debug('The feature_tables list is in the following format.\n')
		
        for i in range(len(indices)):
            logging.debug(f' {i}- { indices[i], feature_tables[i].shape}.')
            
        logging.debug(f" 19- ('Country, Year and Life expectancy'), {feature_tables[-1].shape}.\n")
        
        # Modifying the NaN values by interpolation and mean
        countries = df['Country'].unique()
        modified = NaN.copy()
        
        for i in countries:
            modified[modified['Country'] == i] = NaN[NaN['Country'] == i].interpolate(method = 'linear', limit_direction='both')
        
        isNaN1 = modified.isnull().values.any()
        logging.debug(f' After interpolation, modified still contains NaN values: {isNaN1}')
        logging.debug(f' These NaN values are the ones that could not be interpolated, the whole column for the specific country\'s feature was NaN.')
        logging.debug(f' For this type of NaN value, we have no info, so they will be filled with the column averages of the whole data.\n')
        
        means = NaN.copy()
        means = means.mean()
        
        for i in means.index:
            modified[i] = modified[i].fillna(means[i])
           
        isNaN2 = modified.isnull().values.any()
        logging.debug(f' After this modification checking for NaN values again:')
        logging.debug(f' modified still contains NaN values: {isNaN2}')
        logging.debug(f' Use create_csvs() function to create csv files from the variables in the code.\n')
 
    
        modified = modified.reset_index(drop=True)
        NaN = NaN.reset_index(drop=True)
        NoNaN = NoNaN.reset_index(drop=True)
    
        self.__NoNaN = NoNaN
        self.__NaN = NaN
        self.__modified = modified
        self.__feature_tables = feature_tables
        self.indices = indices
    
    def create_csvs(self):
        '''
        Creates csv files from the variables of the object.
        csv files created are:
        output/NaN.csv
        output/NoNaN.csv
        output/modified.csv
        output/feature_tables/<feature_name>.csv
        '''
        
        assert False == os.path.isdir('output'),'Please erase the current output directory "./output".'
        
        os.mkdir('output')
        self.__NaN.to_csv('output/NaN.csv')
        self.__NoNaN.to_csv('output/NoNaN.csv')
        self.__modified.to_csv('output/modified.csv')
        
        os.mkdir('output/feature_tables')
        for i in range(len(self.indices)):
            self.__feature_tables[i].to_csv(f'output/feature_tables/{i}-{self.indices[i].replace("/", "-")}.csv')
        
        self.__feature_tables[-1].to_csv(f'output/feature_tables/19-Country_Year_LifeExpectancy.csv')
        
        logging.debug(f' csv files created. \n')
        
    @property
    def NoNaN(self):
        return self.__NoNaN.copy()
    	
    @property
    def NaN(self):
        return self.__NaN.copy()
    
    @property
    def modified(self):
        return self.__modified.copy()
    
    @property
    def feature_tables(self):
        return self.__feature_tables.copy()