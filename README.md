# Final-Project

Predicting Spatial and Daily Pattern of Theft from Vehicle in the City of Vancouver 


Libraries
Running the code in this repository requires the installation of:  
	Python 3
	numpy  
	matplotlib 
	seaborn 
	pandas 
	sklearn
	utm
	datetime
	statsmodels
	mpl_toolkits.basemap

Order of Executinon
Cleaning Data ---cleaningdata.py --input crime_csv_all_years.csv --output crime.csv
Data Visualization/Analysis-- crimeanalysis.py --input crime.csv
Data prepration for ML --location.py --input crime.csv --output location.csv
                       --location_removal.py --input crime.csv -- output--no_location.csv


To execute crimeNeighbourhood.py, please use

 python3 crimeNeighbourhood.py location.csv no_location.csv neighbourhood.cvs
