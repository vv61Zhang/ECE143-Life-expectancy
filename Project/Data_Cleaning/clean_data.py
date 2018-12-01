#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 2018
@author: Arda C. Bati
Does data cleaning on the WHO Life expectancy data. Generates different cleaned 
versions of the data in the directory ./output.

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
    
    def __init__(self, df, log_level=logging.INFO):
        '''
        Does data cleaning on the dataset obtained from 'Life Expectancy Data.csv'
        which should be in the same folder with where this function is ran. Contains
        the variables given below:
        
        NoNaN: (DataFrame)
            The version of the dataset with rows containing any NaN value removed.
        NaN: (DataFrame)
            The version of the dataset with 0 values replaced with NaN
        modified:(DataFrame)
            The version of the dataset where NaN values are filled with interpolation / means.
        feature_tables: (list)
            list containing seperate dataframes for each frame from the data.
            
        param: df
            DataFrame input from the extract_data() function from GDP_Pop_Extraction.extraction
        
        param: log_level
            chooses the logging level of the initializer
        '''
        
        self.__NoNaN = 0
        self.__NaN = 0
        self.__modified = 0
        self.__feature_tables = 0
        
        assert isinstance(log_level,int)
        assert 0 <= log_level <= 50
        
        # Logging startup
        logging.basicConfig(stream=sys.stdout, level=log_level)
        #format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        assert isinstance(df,pd.DataFrame)   
               
        df.loc[df['Status'] == 'Developing','Status'] = 1
        df.loc[df['Status'] == 'Developed','Status'] = 2

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
        
        feature_tables = self.__generate_tables(df)
        
        # Modifying the NaN values by interpolation and mean
        to_modify = NaN.copy()
        modified = self.__modify_data(to_modify,NaN)
        
        #Decimal point correction (the data set has decimal point errors)
        sensitivity = 3
        modified = self.__decimal_fix(modified.copy(),sensitivity)
        
        modified = self.__GDP_Pop_Fix(modified.copy())
            
        #Doing the final corrections on the data
        modified, NaN, NoNaN, feature_tables = self.__final_corrections(modified.copy(), NaN.copy(), NoNaN.copy(), feature_tables)

        self.__NoNaN = NoNaN
        self.__NaN = NaN
        self.__modified = modified
        self.__feature_tables = feature_tables
        
        
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
    
    
    @staticmethod
    def __GDP_Pop_Fix(df):
        '''
        This function turns the GDP column of the data frame into GDP per capita and
        population column of the data frame into population (millions)
        
        param:
            df (DataFrame)
        returns:
            result (DataFrame)
        '''
        assert isinstance(df,pd.DataFrame)
        #Modifiyng the GDP as GDP per capita
        df['GDP'] = df['GDP'] / df['Population']
            
        #Representing the population values in millions
        df['Population'] = df['Population'] / 10**6
        result = df
        return result
        
    @staticmethod
    def __modify_data(to_modify,NaN):
        '''
        This is a private static method creates modifies the input DataFrame by interpolation and taking means.
        The aim of this method is to fill all NaN values in the dataset, with reasonable values. The process
        is explained in detail by logging statements.
        
        params:
            to_modify (DataFrame)
        
        returns:
            modified (DataFrame)
        '''
        
        assert isinstance(to_modify,pd.DataFrame)
        assert isinstance(NaN,pd.DataFrame)
        
        countries = to_modify['Country'].unique()
        
        for i in countries:
            to_modify[to_modify['Country'] == i] = NaN[NaN['Country'] == i].interpolate(method = 'linear', limit_direction='both')
        
        isNaN1 = to_modify.isnull().values.any()
        logging.debug(f' After interpolation, modified still contains NaN values: {isNaN1}')
        logging.debug(f' These NaN values are the ones that could not be interpolated, the whole column for the specific country\'s feature was NaN.')
        logging.debug(f' For this type of NaN value, we have no info, so they will be filled with the column averages of the whole data.\n')
        
        means = NaN.copy()
        means = means.mean()
        
        for i in means.index:
            to_modify[i] = to_modify[i].fillna(means[i])
                     
        isNaN2 = to_modify.isnull().values.any()
        logging.debug(f' After this modification checking for NaN values again:')
        logging.debug(f' modified still contains NaN values: {isNaN2}')
        logging.debug(f' Use create_csvs() function to create csv files from the variables in the code.\n')
        
        modified = to_modify
        
        return modified
        

    @staticmethod
    def __generate_tables(df):
        '''
        This is a private static method creates the feature tables for the relevant features of the dataset.
        The logging statements state the details of the process.
        
        params:
            df (DataFrame)
        
        returns:
            feature_tables (list)
        ''' 
        
        assert isinstance(df,pd.DataFrame)
        
        # Going over each feature in the dataset, dropping rows containing NaN
        columns = list(df.columns.values)
        indices = [columns[2]] + columns[4:]
        feature_tables = [df[['Country','Year','Life expectancy ', i]].dropna(0)  for i in indices]
        feature_tables.append(df[['Country','Year','Life expectancy ']].dropna(0))
        
        logging.debug('The feature_tables list is in the following format.\n')
		
        for i in range(len(indices)):
            logging.debug(f' {i}- { indices[i], feature_tables[i].shape}.')
            
        logging.debug(f" 19- ('Country, Year and Life expectancy'), {feature_tables[-1].shape}.\n")
        
        return feature_tables
        
        
    @staticmethod
    def __final_corrections(modified, NaN, NoNaN, feature_tables):
        '''
        This is a private static method which does the final corrections on the CleanData variables.
        It takes the below variables as input, and returns their final corrected versions.
        
        params:
            modified, (DataFrame)
            NaN (DataFrame)
            NoNaN (DataFrame)
            feature_tables (list)
        
        returns:
            modified, (DataFrame)
            NaN (DataFrame)
            NoNaN (DataFrame)
            feature_tables (list)
        '''
        
        assert isinstance(modified,pd.DataFrame)
        assert isinstance(NaN,pd.DataFrame)
        assert isinstance(NoNaN,pd.DataFrame)
        assert isinstance(feature_tables,list)
        
        #Fixing indexing issues
        columns = list(modified.columns.values)
        modified = modified.reset_index(drop=True)
        NaN = NaN.reset_index(drop=True)
        NoNaN = NoNaN.reset_index(drop=True)
        
        columns2 = columns
        columns2.remove('Country')
        
        #Fixing reducing unnecessary precision
        modified[columns2] = modified[columns2].astype(np.float32)
        NaN[columns2] = NaN[columns2].astype(np.float32)
        NoNaN[columns2] = NoNaN[columns2].astype(np.float32)
        
        assert modified[modified == np.inf].sum().sum() != np.inf
        assert NaN[modified == np.inf].sum().sum() != np.inf
        assert NoNaN[modified == np.inf].sum().sum() != np.inf
        
        for table in feature_tables:
            columns_t = list(table.columns.values)
            columns_t.remove('Country')
            table[columns_t] = table[columns_t].astype(np.float32)
            table.reset_index(drop=True)
        
        return modified, NaN, NoNaN, feature_tables
    
    @staticmethod
    def __decimal_fix(df,sensitivity):
        '''
        This is a private static method is used to correct the decimal point errors in the dataset.
        We realized that there are sudden spikes in data through the years, which is mainly caused
        by wrongly placed decimal points. This method tries to fix this issue as much as possible.
        
        sensitivity is an integer value in the open interval (1, 10). It determines the sensitivity
        for detecting the sudden spikes in the dataset, which are caused by decimal point errors.
        
        input df should contain no 0, NaN, or negative values
        
        params:
            df (DataFrame)
            sensitivity (int)
        
        returns:
            df_fixed (DataFrame)
        '''
        
        import numpy as np
        
        assert isinstance(sensitivity,int)
        assert 1 < sensitivity < 10

        assert isinstance(df,pd.DataFrame)
        df = df.replace(0,np.NaN)
        assert df.isnull().values.any() == False
        assert all(df > 0)
        
        columns = list(df.columns.values)
        indices = [columns[2]] + columns[4:]
        
        countries = df['Country'].unique()
        
        means_c = []
        for country in countries:
            means_c.append(df[df['Country'] == country].mean())
            
        indices.remove('Status')
            
        for count,country in enumerate(countries):
            
            for index in indices:
               a2 = df.loc[df['Country'] == country,index]
               
               for loc1,point1 in enumerate(a2):
                   
                   key1 = df.index[df['Country']==country].tolist()
                   key1 = key1[loc1]
                   point1 = df.at[key1,index]
                   
                   if loc1  == 0:
                       point2 = means_c[count]
                       point2 = point2[index]
                   else:
                       key2 = df.index[df['Country']==country].tolist()
                       key2 = key2[loc1-1]
                       point2 = df.at[key2,index]
                       
                   if (((point1 / (point2)) < 1/sensitivity)):
                       while ((point1 / (point2)) < 1/sensitivity):
                           point1 = point1*10
                   elif ((point1 / (point2)) > sensitivity):
                       while ((point1 / (point2)) > sensitivity):
                           point1 = point1/10
                           
                   df.at[key1,index] = point1
        df_fixed = df          
        return df_fixed      
            
    
    def create_csvs(self):
        '''
        Creates csv files from the variables of the object.
        csv files created are:
        output/NaN.csv
        output/NoNaN.csv
        output/modified.csv
        output/feature_tables/<feature_name>.csv
        '''
        
        columns = list(self.NaN.columns.values)
        indices = [columns[2]] + columns[4:]
        
        assert False == os.path.isdir('output'),'Please erase the current output directory "./output".'
        
        os.mkdir('output')
        self.__NaN.to_csv('output/NaN.csv')
        self.__NoNaN.to_csv('output/NoNaN.csv')
        self.__modified.to_csv('output/modified.csv')
        
        os.mkdir('output/feature_tables')
        for i in range(len(indices)):
            self.__feature_tables[i].to_csv(f'output/feature_tables/{i}-{indices[i].replace("/", "-")}.csv')
        
        self.__feature_tables[-1].to_csv(f'output/feature_tables/19-Country_Year_LifeExpectancy.csv')
        
        logging.debug(f' csv files created. \n')
