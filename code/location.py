

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')




#Crime Data
crime = pd.read_csv('crime.csv')




crime = crime[crime['DATE'] != 2017]





labeled = crime[crime.columns.drop('DATE')]
labeled = labeled[labeled.columns.drop('TYPE')]
labeled = labeled[labeled.columns.drop('DAY_WK')]
labeled = labeled[labeled.columns.drop('X')]
labeled = labeled[labeled.columns.drop('Y')]
labeled = labeled[labeled.columns.drop('LAT')]
labeled = labeled[labeled.columns.drop('LONG')]




labeled=labeled.groupby(['NEIGHBOURHOOD','YEAR']).mean().add_suffix('_avg').reset_index()




labeled.to_csv('location.csv', index =False)

