import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sys
from getDictionary import InputVector
from testMNB import MNBs


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], parse_dates=['time_start'])
    
    mnbs = MNBs(df, ['ports', 'dns', 'cs'], 'mac')

    clf = RandomForestClassifier(n_estimators=20, n_jobs=50)

    weeks = [list(range(44,53)), list(range(1,10)), list(range(10,19))]

    #trainCon = df['time_start'] >= pd.Timestamp('2020-03-01')
    trainCon = df['time_start'].dt.week.isin(weeks[2]) 
    #trainCon = df['time_start'] < pd.Timestamp('2020-01-01')
    #testCon = df['time_start'] > pd.Timestamp('2020-02-29')
    testCons = pd.date_range('2019-11-01', '2020-05-01', freq = '1W').tolist()
    #print(testCons)

    #mnbs.extractDictionary(trainCon)
    mnbs.extractDictionary(df['time_start'] < pd.Timestamp('2021-01-01'))
    mnbs.fit(trainCon)

    trainDF = mnbs.updateDFWithProba(trainCon)

    #print(trainDF)
    
    features = ['volume_mean','flow_durations','ratio','sleep_time','dns_interval','ports_id','ports_proba','dns_id','dns_proba','cs_id','cs_proba']
    X = trainDF[features]
    y = trainDF['mac']

    clf.fit(X, y)
   
    for date in testCons:
        testCon = df['time_start'].dt.week == pd.Timestamp(date).week
        testDF = mnbs.updateDFWithProba(testCon)

        X = testDF[features]
        y = testDF['mac']

        y_pred = clf.predict(X)
        print(metrics.classification_report(y, y_pred))
        print(f"Accuracy {date.week}:", metrics.accuracy_score(y, y_pred), metrics.f1_score(y, y_pred, average="micro"))


