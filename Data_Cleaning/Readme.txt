Written by Arda C. Bati, abati@ucsd.edu

Project_Data.py script does cleaning and structuring on the dataset of our project, named "Life Expectancy Data.csv." It was developed in Spyder, so if there are any module problems etc. please tell me.

Link to the dataset: https://www.kaggle.com/kumarajarshi/life-expectancy-who/home

Project_Data.py  and "Life Expectancy Data.csv" should be on the same folder for the script to work. (There is an assertion for this.)

Most of the explanation is in the code as logging messages. Therefore I will not repeat them here to prevent confusion. To get output from the script, create_csv() function should be called. This will create an "./output" directory and generate corresponding csv files in there. Output directory should be erased each time create_csv() is called (there is an assertion for this.)

Please tell me any questions/problems about the code. As far as I checked, the output is as expected. I didn't check rigorously for bugs, I just put relevant assertions about filenames. 

Also please tell me if we can do anything else for cleaning the code. I didn't do any predictive type of cleaning. I either removed the NaN values or filled them with interpolation and mean.
