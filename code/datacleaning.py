
import pandas as pd
import numpy as np
import utm
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')




#Crime Data
crime = pd.read_csv('crime_csv_all_years.csv',header = 0, parse_dates=[1])

# Cleaning the data. Excluding columns that will not be used in analysis. Puting data in correct formate.


#EXclude Hundered_BLock
crime.drop(['HUNDRED_BLOCK'],axis =1, inplace = True)


#Excluded Minute
crime.drop(['MINUTE'], axis = 1, inplace=True)

#Change formating of year
crime['YEAR'] = pd.to_datetime(crime['YEAR'], format='%Y').dt.year

# The year 2018 will be excluded from analysis since there is still 5 months left in the year

crime = crime[crime['YEAR'] != 2018]


#HOUR, MINUTE, HUNDRED_BLOCK and NEIGHBOURHOOD are missing data. Exclude from analysis
crime.dropna()

# convert it to a datetime data type
crime['DATE'] = pd.to_datetime({'YEAR':crime['YEAR'], 'MONTH':crime['MONTH'], 'DAY':crime['DAY']})

#Create Day of Week column for analysis (Monday=0 to Sunday=6)
crime['DAY_WK'] = crime['DATE'].dt.dayofweek


# Plot a Bar graph of the data 

sns.set(style = 'darkgrid')
plt.figure(figsize=(8,8))
allcrime = crime.groupby(crime['TYPE']).size().sort_values(ascending=True)
allcrime.plot(kind='barh', color ='b')
plt.title('City of Vancouver Crime by Type 2003 to 2017', size =18)
plt.ylabel('Type of Crime', size =16)
plt.xlabel('Number of Crimes',size =16, wrap=True)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.savefig('TotalCrimebyType.png')
plt.show()



# Has the rate of vehicle offenses changed over the years
yearly_crime=crime.groupby([crime["YEAR"]]).size().to_frame().reset_index().rename(columns={0:"Count"})




plt.figure(figsize =(8,10)) 
yearly_crime.plot(kind = "bar", x=['YEAR'], y = "Count",color = "blue", linewidth = 8, legend = False)
plt.title('City of Vancouver Crime by Year', size =18)
plt.ylabel('Number of Crimes', size =16)
plt.xlabel('Year',size =16)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.tight_layout()
plt.savefig('TotalCrimebyYear.png',dpi=100,alpha=True)
plt.show()


# The focus of the analysis will be theft from vehicle since it is the most frequent type of crime. 



# Interested only in Theft from Vehicle
crime = crime[crime['TYPE']=='Theft from Vehicle']




# X and Y coordinates are in UTM 10 WGS84 formate (X: Easting, Y Northing)(NOTE: cannot have rows with zero)
# This will be useful in prediction analysis


#Convert x and Y coordinates to Lon & Lat 
#Adapted from: http://www.worthandlung.de/pandas/Converting-e32-n32-to-lat-long.html
def getUTMs(row):
    tup = utm.to_latlon(row.ix[0],row.ix[1], 10, 'U')
    return pd.Series(tup[:2])



crime[['LAT','LONG']] = crime[['X','Y']].apply(getUTMs , axis=1)
crime.head()


# Data is now ready for analysis. Export to crime.csv


crime.to_csv('crime.csv', index =False)

