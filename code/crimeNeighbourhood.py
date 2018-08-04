import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def bayes_model(X_train, y_train) :
    model = make_pipeline(StandardScaler(), GaussianNB(priors=None))
    model.fit(X_train,y_train)
    return model

def knn_model(X_train, y_train) :
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors= 5))
    model.fit(X_train, y_train)
    return model

def svc_model (X_train, y_train) :
    model = make_pipeline(StandardScaler(), SVC(kernel='linear', decision_function_shape='ovr'))
    model.fit(X_train, y_train)  
    return model

def test_model (unlabelled) :
    test = unlabelled.drop(['YEAR','NEIGHBOURHOOD'], axis = 1).values
    return test


def main() :
    labelled = pd.read_csv(sys.argv[1])
    #print (labelled)
    unlabelled = pd.read_csv(sys.argv[2])
    labeled = labelled.drop(['YEAR'], axis =1)
    X = labeled.drop(['NEIGHBOURHOOD'], axis =1).values
    y = labeled['NEIGHBOURHOOD'].values 
    #print (unlabelled)
    #X, y = get_train_data(labelled)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #model = bayes_model(X_train, y_train)#Training score: 0.281481, Testing score: 0.155556
    model = knn_model(X_train, y_train)# Training score: 0.348148,Testing score: 0.111111
    #model = svc_model(X_train, y_train) #0.214815, Testing score: 0.0222222
    test_data = test_model(unlabelled)
    predictions = model.predict(test_data)
    df = pd.DataFrame({'truth': y_test, 'prediction': model.predict(X_test)})
    #print(df[df['truth'] != df['prediction']])
    print("Training score: %g\nTesting score: %g" % (model.score(X_train, y_train), model.score(X_test, y_test)))
    pd.Series(predictions).to_csv(sys.argv[3], index=False)
    


if __name__ == '__main__':
    main()
