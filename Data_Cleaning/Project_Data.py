"""
Created on Mon Nov 12 2018

@author: Arda C. Bati

Does data cleaning on the WHO Life expectancy data. Generates different cleaned 
versions of the data in the directory ./output. This script should be in the same
folder/directory with the file 'Life Expectancy Data.csv', which is provided on 
Group 5's github page. 

After running the "Project_data.py" script, use the create_csv() function to create
.csv files from the variables obtained in the code. The code is completely explained
by logging statements. Logging level should changed to remove the these messages.
"""

import pandas as pd
import numpy as np
import logging, sys
import os

# Logging startup

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Analyzing the data

logging.debug(f' ***************************')
logging.debug(f' PART 1) ANALYZING THE DATA')
logging.debug(f' ***************************\n')

assert os.path.isfile('Life Expectancy Data.csv'),'Please copy "Life Expectancy Data.csv" to the current directory.'
    
df = pd.read_csv('Life Expectancy Data.csv')
columns = list(df.columns.values)
logging.debug(f'\nThe column labels of the data are : {columns}\n')
logging.debug(f' Shape of the data: {df.shape}\n')

#total_size = df.size
#column_count = len(df.columns); 
#row_count = len(df.index)
#year_count = len(df['Year'].unique())
#country_count = len(df['Country'].unique())
  
countries_years = df.groupby('Country').count()['Year']
countries_missing_years = countries_years[countries_years != 16]
cmy_count = len(countries_missing_years)

logging.debug(f' {cmy_count} countries have input for only 1 year instead of 15 \n')
logging.debug(countries_missing_years)
logging.debug('\n')
logging.debug(' These countries will be removed from the dataset as we don\'t have enough information about them. \n')

for i in countries_missing_years.index:
    df = df[df.Country != i]
logging.debug(' Above countries are removed')    
    
logging.debug(f' ***********************************************')
logging.debug(f' PART 2) DROPPING ALL ROWS WITH NAN VALUES')
logging.debug(f' ***********************************************\n')

# Replacing 0 values with NaN

logging.debug(' In the dataset, unkown values are filled as NaN or 0.')
logging.debug(' I am replacing all 0 values with NaN.')
logging.debug(' df_NaN in the code contains the above mentioned form of the data.\n')
df_NaN = df.replace(0,np.nan)
df_NaN_Original = df_NaN.copy()

# Dropping NaN rows

# Dropping rows containing NaN in any of their columns
logging.debug(' Dropping all rows that contain NaN from df_NaN.')
logging.debug(' This new version is named df_NoNaN.\n')
df_NoNaN = df_NaN.copy()
df_NoNaN = df_NoNaN.dropna(0)

logging.debug(f' ***********************************************')
logging.debug(f' PART 3) FEATURE BY FEATURE, DROPPING NAN VALUES')
logging.debug(f' ***********************************************\n')

# Going over each feature in the dataset, dropping rows containing NaN
logging.debug(' Going over each feature in the dataset (df in code) seperately, dropping rows containing NaN.')
indices = [columns[2]] + columns[4:]
cleaned_features = [df[['Country','Year','Life expectancy ', i]].dropna(0)  for i in indices]
cleaned_features.append(df[['Country','Year','Life expectancy ']].dropna(0))
logging.debug(' The result is cleaned_features in the code, and it\'s format is as follows (feature,(rows,columns)):')
logging.debug(' (The 4 columns include Country, Year, Life expectancy, and the expressed feature).\n')

for i in range(len(indices)):
    logging.debug(f' {i}- { indices[i], cleaned_features[i].shape}.')
    
logging.debug(f" 19- ('Country, Year and Life expectancy'), {cleaned_features[-1].shape} :\n")

# Modifying the NaN values by interpolation and mean

logging.debug(f' *******************************************')
logging.debug(f' PART 4) MODIFYING BY INTERPOLATION AND MEAN')
logging.debug(f' *******************************************\n')


logging.debug(' Modifying the NaN values of df_NaN by interpolation and mean.')
countries = df['Country'].unique()
df_modified = df_NaN.copy()

logging.debug(' First I do interpolation on df_NaN, country by country and feature by feature.')
logging.debug(' The result is stored in df_modified.\n')

for i in countries:
    df_modified[df_modified['Country'] == i] = df_NaN[df_NaN['Country'] == i].interpolate(method = 'linear', limit_direction='both')

isNaN1 = df_modified.isnull().values.any()
logging.debug(f' After interpolation, df_modified still contains NaN values: {isNaN1}')
logging.debug(f' These NaN values are the ones that could not be interpolated, the whole column for the specific country\'s feature was NaN.')
logging.debug(f' For this type of NaN value, we have no info, so they will be filled with the column averages of the whole data.\n')

means = df_NaN.copy()
means = means.mean()

for i in means.index:
    df_modified[i] = df_modified[i].fillna(means[i])
    
isNaN2 = df_modified.isnull().values.any()
logging.debug(f' After this modification checking for NaN values again:')
logging.debug(f' df_modified still contains NaN values: {isNaN2}')
logging.debug(f' df_modified shape is {df_modified.shape}\n')
logging.debug(f' We should not use df_modified for making very specific inferences about countries or smaller regions. (As it contains modified values)')
logging.debug(f' As an example, "Total expenditure" of USA was all NaN\'s so it was filled with the general average.\n')

logging.debug(f' Use create_csv() function to create csv files from the variables in the code.\n')

def create_csv():
    '''
    Creates csv files from the variables in the above code.
    csv files created are:
    output/df_NaN.csv
    output/df_NoNaN.csv
    output/df_modified.csv
    output/features/<feature_name>.csv
    '''
    
    assert False == os.path.isfile('Life Expectancy Data.csv'),'Please erase the current output directory "./output".'
    
    os.mkdir('output')
    df_NaN_Original.to_csv('output/df_NaN.csv')
    df_NoNaN.to_csv('output/df_NoNaN.csv')
    df_modified.to_csv('output/df_modified.csv')
    
    os.mkdir('output/features')
    for i in range(len(indices)):
        cleaned_features[i].to_csv(f'output/features/{i}-{indices[i].replace("/", "-")}.csv')
    
    cleaned_features[-1].to_csv(f'output/features/19-Country_Year_LifeExpectancy.csv')

    logging.debug(f' csv files created. \n')
