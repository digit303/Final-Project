

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')




#Crime Data
crime = pd.read_csv('crime.csv')





crime.head()


# Exlcuding all location identifiers to train the prediction model to predict Neighbourhood



crime = crime[crime['YEAR']==2017]
#crime.head(30)



area=crime.groupby([crime["NEIGHBOURHOOD"]]).size().to_frame().reset_index().rename(columns={0:"Count"})
area.head()           



unlabeled = crime[crime.columns.drop('X')]
unlabeled = unlabeled[unlabeled.columns.drop('Y')]
unlabeled = unlabeled[unlabeled.columns.drop('LAT')]
unlabeled = unlabeled[unlabeled.columns.drop('LONG')]
unlabeled = unlabeled[unlabeled.columns.drop('DATE')]
unlabeled = unlabeled[unlabeled.columns.drop('TYPE')]
unlabeled = unlabeled[unlabeled.columns.drop('DAY_WK')]





unlabeled.head(20)





unlabeled=unlabeled.groupby(['NEIGHBOURHOOD','YEAR']).mean().add_suffix('_avg').reset_index()
unlabeled.head(23)





unlabeled.to_csv('no_location.csv', index =False)

