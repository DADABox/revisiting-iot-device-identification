import pandas as pd
import numpy as np
import ast, sys
from collections import Counter

class InputVector(object):

    def __init__(self):
        self.input = {}
        self.wholeDict = []


    def extractDataFromDF(self, df, indexCol, dataCol):
        for index, rec in df[(df[dataCol] != "{}")].groupby(indexCol)[dataCol].unique().iteritems():
            self.input[index] = self._extractData(rec)
            self.wholeDict += self.input[index]

        self.wholeDict = list(set(self.wholeDict))

    def _extractData(self, data):
        flatten = lambda t: [item for sublist in t for item in sublist]
        return list(set(flatten([list(ast.literal_eval(x).keys()) for x in data])))

    def _getEmptyInputVector(self, deviceId = None):
        input = self.wholeDict if deviceId is None else self.input[deviceId]

        vector = dict(zip(input, [0]*len(input)))
        vector['other'] = 0
        return vector
    
    def getFinalInputVector(self, inputData: dict, deviceId = None):
        vector = self._getEmptyInputVector(deviceId)
        for k, v in inputData.items():
            if k in vector:
                vector[k] = v
            else:
                vector['other'] += v

        #return vector
        return list(vector.values())

    def getFinalInputMatrix(self, df: pd.DataFrame, dataCol: str):
        X = []
        for index, rec in df[dataCol].iteritems():
            X.append(self.getFinalInputVector(ast.literal_eval(rec)))

        return X

    def getXy(self, df: pd.DataFrame, dataCol: str, indexCol: str):
        X = self.getFinalInputMatrix(df, dataCol)
        y = df[indexCol]

        return np.array(X), np.array(y)
        

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], parse_dates=['time_start'])
    dns = InputVector()
    dns.extractDataFromDF(df[df['time_start'] < pd.Timestamp("2020-03-01")], 'mac', 'dns')
    #print(dns.input)
    mar = df[df['time_start'] > pd.Timestamp("2020-02-29")]
    X,y = dns.getXy(mar, 'dns', 'mac')
    print(X)
    print(y)
    #print(dns.getFinalInputVector(ast.literal_eval(mar.iloc[20750]['dns'])))#, mar.iloc[20750]['mac']))

