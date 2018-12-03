# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:07:09 2018

@author: Weinan(Eric) Li. w3li@ucsd.edu
"""

#requires pygal,pygal_maps_world,pandas,numpy

import pygal.maps.world
from IPython.display import SVG
from Project.Data_Cleaning import clean_data
import pygal.style
from pygal.style import Style
import numpy as np
import pandas as pd
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



def display_Expectancy(data,feature,year):
    """
    Show life expectancy of all countries in data on map.
    
    input:
        data, panda.dataframe
        feature, str, Life expectancy
        year, int, given year
    output:
        map
    """
    from pygal_maps_world.i18n import COUNTRIES
    assert isinstance(data,pd.core.frame.DataFrame)
    assert isinstance(feature,str) and feature=='Life expectancy '
    assert isinstance(year,int) and year<2016 and year>1999
    
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

    
    #divide data into groups
    lifedata=pd.DataFrame.from_dict(lifedata,orient='index',columns=['Value'])
    life40=lifedata[(lifedata['Value']<40) ]
    life50=lifedata[(lifedata['Value']>=40) & (lifedata['Value']<50) ]
    life60=lifedata[(lifedata['Value']>=50) & (lifedata['Value']<60) ]
    life70=lifedata[(lifedata['Value']>=60) & (lifedata['Value']<70) ]
    life80=lifedata[(lifedata['Value']>=70) & (lifedata['Value']<80) ]
    life90=lifedata[(lifedata['Value']>=80) & (lifedata['Value']<90) ]
    
    color1=(255,1,1)
    color2=(1,200,255)
    
    fractionlist=list(np.linspace(0,1,6))
    colorlist=list()
    for i in fractionlist:
        rgb=blend_color(color1,color2,i)
        hexa=decimal2hex(rgb)
        colorlist.append(hexa)
    

    #build map style
    custom_style = Style(
  background='transparent',
  plot_background='transparent',
  foreground='#53E89B',
  foreground_strong='#53A0E8',
  foreground_subtle='#630C0D',
  opacity='.6',
  opacity_hover='.9',
  transition='400ms ease-in',
  colors=tuple(colorlist))
    
        
    
    
    
    worldmap_chart = pygal.maps.world.World(style=custom_style)
    worldmap_chart.title = feature+' in a given year'
    
    worldmap_chart.add('<40', life40.to_dict()['Value'])
    worldmap_chart.add('<50', life50.to_dict()['Value'])
    worldmap_chart.add('<60', life60.to_dict()['Value'])
    worldmap_chart.add('<70', life70.to_dict()['Value'])
    worldmap_chart.add('<80', life80.to_dict()['Value'])
    worldmap_chart.add('<90', life90.to_dict()['Value'])
    
    #worldmap_chart.add('<40', lifedata.to_dict()['Value'])
    
    return SVG(worldmap_chart.render())




def blend_color(color1, color2, f):
    """
    find gradual color between color1 and color2, f as fraction
    """
    assert isinstance(color1,tuple)
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    r = r1 + (r2 - r1) * f
    g = g1 + (g2 - g1) * f
    b = b1 + (b2 - b1) * f
    return r, g, b
def decimal2hex(color):
    """
    convert decimal rgb color to hex
    """
    ans='#'
    for i in color:
        if len(hex(int(i))[2:])<2:
            ans+='0'+hex(int(i))[2:]
        else:
            ans+=hex(int(i))[2:]
        
    
    
    return ans
