import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import sys
from sklearn import metrics
from joblib import dump, load
from pathlib import Path

features = ['srcPort', 'destPort',
       'bytes_out', 'num_pkts_out', 'bytes_in', 'num_pkts_in', 'f_ipt_mean',
       'f_ipt_std', 'f_ipt_var', 'f_ipt_skew', 'f_ipt_kurtosis', 'f_b_mean',
       'f_b_std', 'f_b_var', 'f_b_skew', 'f_b_kurtosis', 'duration', 'pr',
       'domainId']
label = 'deviceId'

fieldTypes = {'deviceId': 'int16', 
        #'time_start': 'datetime64[ns]', 'time_end': 'datetime64[ns]', 
        'srcPort': 'int32', 'destPort': 'int32',
        'bytes_out': 'int32', 'num_pkts_out': 'int32',
        'bytes_in': 'int32', 'num_pkts_in': 'int32',
        'f_ipt_mean': 'float64', 'f_ipt_std': 'float64',
        'f_ipt_var': 'float64', 'f_ipt_skew': 'float64',
        'f_ipt_kurtosis': 'float64',
        'f_b_mean': 'float64', 'f_b_std': 'float64',
        'f_b_var': 'float64', 'f_b_skew': 'float64',
        'f_b_kurtosis': 'float64',
        'duration': 'float64', 'pr': 'int8',
        'expire_type': 'object',
        'domain': 'object',
        'domain2': 'object',
        'domainId': 'int16'}


#df = pd.read_csv(sys.argv[1], dtype = fieldTypes, parse_dates = ['time_start'])
df = pd.read_csv(sys.argv[1], parse_dates = ['time_start'])
df.fillna(0, inplace=True)

weeks = [list(range(44,53)), list(range(1,10)), list(range(10,19))]

#trainCon = df['time'] >= pd.Timestamp('2020-03-01')
#trainCon = df['time'] < pd.Timestamp('2020-01-01')
#trainCon = ((df['time'] >= pd.Timestamp('2020-01-01')) & (df['time'] < pd.Timestamp('2020-03-01')))
trainCon = df['time_start'].dt.week.isin(weeks[int(sys.argv[2])])
#trainCon = df['time'].dt.week.isin([44,45])


#testCon = df['time'] > pd.Timestamp('2020-02-29')
#testCons = pd.date_range('2019-11-01', '2020-05-01', freq = '1W').tolist()
testCons = pd.date_range('2019-12-01', '2020-05-01', freq = '1W').tolist()
#testCons = pd.date_range('2019-11-01', '2019-12-01', freq = '1W').tolist()

#X = dfTrain.iloc[:, 2:5]
#y = dfTrain.iloc[:, 1]
X = df[trainCon][features]
y = df[trainCon][label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, stratify=y)
#clf = RandomForestClassifier(n_estimators=20, n_jobs=50)
rfc = RandomForestClassifier(n_jobs=-1)
#svc = SVC(verbose=True)
#dtc = DecisionTreeClassifier()
#knn = KNeighborsClassifier(n_jobs=54)
#mv = VotingClassifier(estimators=[('knn', knn), ('dt', dtc), ('rf', rfc), ('svm',svc)], voting='hard', n_jobs=54)

modelFile = Path(f"model/rfc_{sys.argv[2]}")
if modelFile.is_file():
    print(f"loading model {modelFile}")
    rfc = load(modelFile)
else:
    print(f"start training for weeks {weeks[int(sys.argv[2])]}")
    rfc.fit(X_train, y_train)
    print("RFC trained")
    #dump(rfc, f"model/rfc_{sys.argv[2]}")
    dump(rfc, modelFile)

#svc.fit(X, y)
#print("SVC trained")
#dump(SVC, f"model/svc_{sys.argv[2]}")
#
#dtc.fit(X, y)
#print("DTC trained")
#dump(dtc, f"model/dtc_{sys.argv[2]}")
#
#knn.fit(X,y)
#print("KNN Trained")
#dump(knn, f"model/knn_{sys.argv[2]}")
#
#mv.fit(X, y)
#print("MV Trained")
#dump(mv, f"model/mv_{sys.argv[2]}")

for date in testCons:
    testCon = df['time_start'].dt.week == pd.Timestamp(date).week
    #testCon = df['time'].dt.date == pd.Timestamp(date).date
    testDF = df[testCon]

    X = testDF[features]
    y = testDF[label]
    
    if len(X) == 0: 
        continue
    
    #y_pred = clf.predict(X)
    y_rfc = rfc.predict(X)
    #print("rfc predict")
    #y_svc = svc.predict(X)
    #print("svc predict")
    #y_dtc = dtc.predict(X)
    #print("dtc predict")
    #y_knn = knn.predict(X)
    #print("knn predict")
    #y_mv = mv.predict(X)
    #print("mv predict")
    
    print(f"{date.week}", metrics.f1_score(y, y_rfc, average='micro'))
            #metrics.f1_score(y, y_svc, average='micro'),
            #metrics.f1_score(y, y_dtc, average='micro'),
            #metrics.f1_score(y, y_knn, average='micro'),
            #metrics.f1_score(y, y_mv, average='micro')
            #)
    #print(f"Accuracy {date.date}:", metrics.accuracy_score(y, y_pred))

sys.exit()


