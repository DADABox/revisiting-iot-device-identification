import pandas as pd
import ast, sys
import numpy as np
from getDictionary import InputVector
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

class MNB(object):

    def __init__(self, df: pd.DataFrame, dataCol: str, indexCol: str):
        self.df = df
        self.dataCol = dataCol
        self.indexCol = indexCol
        self.clf = MultinomialNB()
        self.iv = InputVector()

    def extractDictionary(self, cond):
        self.iv.extractDataFromDF(self.df[cond], self.indexCol, self.dataCol)

    def fit(self, cond):
        X, y = self.iv.getXy(self.df[cond], self.dataCol, self.indexCol)

        self.clf.fit(X, y)

    def predictProba(self, cond):
        X, y = self.iv.getXy(self.df[cond], self.dataCol, self.indexCol)

        predy = self.clf.predict_proba(X)
        indices = np.argmax(predy, axis=1)
        vals = np.amax(predy, axis=-1)
        
        return np.array(list(zip(indices, vals)))

class MNBs(object):

    def __init__(self, df: pd.DataFrame, dataCol: list, indexCol: str):
        self.df = df
        self.mnb = {}
        for col in dataCol:
            self.mnb[col] = MNB(df, col, indexCol)

    def extractDictionary(self, cond):
        for mnb in self.mnb.values():
            mnb.extractDictionary(cond)

    def fit(self, cond):
        for mnb in self.mnb.values():
            mnb.fit(cond)

    def predictProba(self, cond):
        probs = {}
        for col, mnb in self.mnb.items():
            probs[col] = mnb.predictProba(cond)

        return probs

    def updateDFWithProba(self, cond) -> pd.DataFrame:
        probs = self.predictProba(cond)
        
        df = self.df[cond].copy()

        for col, proba in probs.items():
            df[f'{col}_id'], df[f'{col}_proba'] = proba[:, 0], proba[:, 1]

        return df




if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], parse_dates=['time_start'])

    weeks = [list(range(44,53)), list(range(1,10)), list(range(10,19))]

    mnb = MNB(df, 'dns', 'mac')
    feb = df['time_start'] < pd.Timestamp("2020-03-01")
    mar = df['time_start'] > pd.Timestamp("2020-02-29")
    mnb.extractDictionary(feb)
    mnb.fit(feb)

    pred = mnb.predictProba(mar)
    print(pred)
    sys.exit()
    


    dns = InputVector()
    cs = InputVector()
    
    dnsClf = MultinomialNB()
    csClf = MultinomialNB()

    #dnsClf = RandomForestClassifier(n_estimators=20, n_jobs=50)
    #csClf = RandomForestClassifier(n_estimators=20, n_jobs=50)


    feb = df[df['time_start'] < pd.Timestamp("2020-03-01")]
    mar = df[df['time_start'] > pd.Timestamp("2020-02-29")]

    dns.extractDataFromDF(feb, 'mac', 'dns')
    cs.extractDataFromDF(feb, 'mac', 'cs')
    
    dnsX, dnsy = dns.getXy(feb, 'dns', 'mac')
    csX, csy = cs.getXy(feb, 'cs', 'mac')

    dnsClf.fit(dnsX, dnsy)
    csClf.fit(csX, csy)


    dnsTestX, dnsTesty = dns.getXy(mar, 'dns', 'mac')
    csTestX, csTesty = cs.getXy(mar, 'cs', 'mac')
    
    #print(f"dnsX shape {dnsX.shape} dnsy {dnsy.shape} test x: {dnsTestX.shape} test y: {dnsTesty.shape}")
    #print(f"csX shape {csX.shape} csy {csy.shape} test x: {csTestX.shape} test y: {csTesty.shape}")

    dnsPredicty = dnsClf.predict_proba(dnsTestX)
    indices = np.argmax(dnsPredicty, axis=1)
    vals = np.amax(dnsPredicty, axis=-1)
    print(np.array(list(zip(indices, vals))))
    sys.exit()
    csPredicty = csClf.predict(csTestX)

    print("Accuracy next month DNS:", metrics.accuracy_score(dnsTesty, dnsPredicty))
    print("Accuracy next month CS:", metrics.accuracy_score(csTesty, csPredicty))
