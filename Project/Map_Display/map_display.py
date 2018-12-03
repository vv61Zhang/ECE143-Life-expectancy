# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:07:09 2018

@author: Weinan(Eric) Li. w3li@ucsd.edu
"""

#requires pygal,pygal_maps_world,pandas,numpy

import pygal.maps.world
from IPython.display import SVG
from Project.Data_Cleaning import clean_data

def display(data,feature,year):
    '''
    Takes a CleanData object "data", a "feature" from this dataset other than Country
    or Year, and a specific year. On a world map, draws how this feature is distributed
    in the given year. The year should not be outside the range of the dataset.
    
    param: data     type: CleanData object
    param: feature  type: str
    param: year     type: int
    '''
    from pygal_maps_world.i18n import COUNTRIES
    
    assert isinstance(data, clean_data.CleanData)
    
    assert isinstance(feature,str)
    valid_features = list(data.modified.columns)
    valid_features.remove('Country')
    valid_features.remove('Year')
    assert feature in valid_features
    
    assert isinstance(year,int)
    modified = data.modified
    assert modified.Year.min() <= year <= modified.Year.max()   
    
    #match the country codes
    countries={value:key for key, value in COUNTRIES.items()}
    countries['United States of America']='us' # there're more needs to manually match
    countries['United States of America']='us' # there're more needs to manually match
    countries['United Kingdom of Great Britain and Northern Ireland']='gb'
    countries['Bolivia (Plurinational State of)']='bo'
    countries["Côte d'Ivoire"] = 'ci'
    countries['Cabo Verde']='cv'
    countries['Czechia']='cz'
    countries["Democratic People's Republic of Korea"]='kp'
    countries['Democratic Republic of the Congo']='cd'
    countries['Iran (Islamic Republic of)']='ir'
    countries['Libya']='ly'
    countries['Republic of Korea']='kr'
    countries['Republic of Moldova']='md'
    countries['The former Yugoslav republic of Macedonia']='mk'
    countries['United Republic of Tanzania']='tz'
    countries['Venezuela (Bolivarian Republic of)']='ve'
    
    
    display_data=dict()
    
    for i in range(modified.shape[0]):
        row=modified.loc[i]
        if row['Year']==year:
            countryname=row['Country']
            display_feature=row[feature]
            try:
                display_data[countries[countryname]]=display_feature
            except:
                pass
    
    #colors need to be adjusted for clearer display
    worldmap_chart = pygal.maps.world.World()
    worldmap_chart.title = '{0} in the year {1}'.format(feature,year)
    worldmap_chart.add('In {0}'.format(year), display_data)
    
    return SVG(worldmap_chart.render())



def displaymap(data,feature,style,year):
    """
    Show life expectancy of all countries in data on map.
    
    input:
        data, panda.dataframe
        feature, str
        style, pygal.style
        year, int, given year
    output:
        map
    """
    
    countries={value:key for key, value in COUNTRIES.items()}
    #manually add countries with different names between api and csv file.
    countries['United States of America']='us' # there're more needs to manually match
    countries['United Kingdom of Great Britain and Northern Ireland']='gb'
    countries['Bolivia (Plurinational State of)']='bo'
    countries["Côte d'Ivoire"] = 'ci'
    countries['Cabo Verde']='cv'
    countries['Czechia']='cz'
    countries["Democratic People's Republic of Korea"]='kp'
    countries['Democratic Republic of the Congo']='cd'
    countries['Iran (Islamic Republic of)']='ir'
    countries['Libya']='ly'
    countries['Republic of Korea']='kr'
    countries['Republic of Moldova']='md'
    countries['The former Yugoslav republic of Macedonia']='mk'
    countries['United Republic of Tanzania']='tz'
    countries['Venezuela (Bolivarian Republic of)']='ve'
    
    
    lifedata=dict()
    #brute force, needs perfection, only mean to show result
    for i in range(data.shape[0]):
        row=data.loc[i]
        if row['Year']==year:
            countryname=row['Country']
            target=row[feature]
            try:
                lifedata[countries[countryname]]=target
            except:
                #print(countryname)
                pass
        
    #build colorbar 
    maxValue=max(lifedata.values())
    minValue=min(lifedata.values())
    
    #colors need to be adjusted for clearer display
    lifedata=pd.DataFrame.from_dict(lifedata,orient='index',columns=['Value'])
    life40=lifedata[(lifedata['Value']<40) ]
    life50=lifedata[(lifedata['Value']>=40) & (lifedata['Value']<50) ]
    life60=lifedata[(lifedata['Value']>=50) & (lifedata['Value']<60) ]
    life70=lifedata[(lifedata['Value']>=60) & (lifedata['Value']<70) ]
    life80=lifedata[(lifedata['Value']>=70) & (lifedata['Value']<80) ]
    life90=lifedata[(lifedata['Value']>=80) & (lifedata['Value']<90) ]
    
    
    worldmap_chart = pygal.maps.world.World(style=style)
    worldmap_chart.title = feature+' in a given year'
    
    worldmap_chart.add('<40', life40.to_dict()['Value'])
    worldmap_chart.add('<50', life50.to_dict()['Value'])
    worldmap_chart.add('<60', life60.to_dict()['Value'])
    worldmap_chart.add('<70', life70.to_dict()['Value'])
    worldmap_chart.add('<80', life80.to_dict()['Value'])
    worldmap_chart.add('<90', life90.to_dict()['Value'])
    
    #worldmap_chart.add('<40', lifedata.to_dict()['Value'])
    
    return SVG(worldmap_chart.render())