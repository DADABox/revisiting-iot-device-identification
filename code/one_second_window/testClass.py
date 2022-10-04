from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

import pandas as pd
import numpy as np
import sys, os
import warnings
import datetime
#import pickle
from joblib import dump, load


warnings.simplefilter("ignore", category=FutureWarning)

class Evaluator(object):

  def __init__(self, modelType = ""):
    self.modelType = modelType
    self.model = None
    self.testFile = ""
    self.trainFile = ""
    self.metrics = {}

    self.modelBaseDir = "model"

    #self.XCols = ['avg', 'std', 'n_bytes']
    #self.label = 'label'

    self.XCols = ['destPort', 'bytes_out', 'bytes_in', 'num_pkts_out', 'num_pkts_in', 
        'f_ipt_mean', 'f_ipt_std', 'f_b_mean', 'f_b_std', 'duration', 'pr']#, 'domain2']
    self.label = 'deviceId'

  def loadModel(self, fileName):
    self.model = load(os.path.join(self.modelBaseDir, fileName))

  def incrementalTrainEval(self, startDate, endDate, step = 1):
    dates = pd.date_range(startDate, endDate, freq = "D").tolist()
    self.model = self.getModel()

    for i, date in enumerate(dates):
      try:
        trainData = self.filterByDate(date, dates[i+1])
        #print(trainData) #print(trainData['time_start'][0], trainData['time_start'][-1])
        testData = self.filterByDate(dates[i+1], dates[i+2])
        #print(testData[0]['time_start'], testData[-1]['time_start'])

        trainX, trainY = self.getXandY(trainData)
        if not trainX.empty:
          #print (trainX, trainY)
          self.model.fit(trainX, trainY)
        
        testX, testY = self.getXandY(testData)
        
        if not testX.empty:
          self.yPredict = self.model.predict(testX)
          self.y = testY
          self.evaluateModel()

      except (IndexError, KeyError):
        print (date, i, len(dates))


      

  def trainModel(self, trainFile):
    self.trainFile = trainFile

    self.model = self.getModel()
    self.loadData(trainFile)
    X, y = self.getXandY()
    #print(X)
    self.model.fit(X, y)

  def testModel(self, testFile):
    self.loadData(testFile)
    X, y = self.getXandY()

    self.yPredict = self.model.predict(self.X)

  def evaluateModel(self):
    self.metrics['acc'] = accuracy_score(self.y, self.yPredict)
    self.metrics['recall'] = recall_score(self.y, self.yPredict, average="micro")
    self.metrics['f1'] = f1_score(self.y, self.yPredict, average="micro")
    self.metrics['spec'] = specificity_score(self.y, self.yPredict, average="micro")
    self.metrics['mean'] = geometric_mean_score(self.y, self.yPredict, average="micro")

    print(self.metrics)

  def loadData(self, fileName):
    #df = pd.read_csv(fileName, names = self.XCols + [self.label], low_memory = False)
    self.df = pd.read_csv(fileName, low_memory = False)
    for col in self.XCols:
      self.df[col].fillna(0, inplace = True)
    
    self.df['time_start'] = self.df['time_start'].apply(lambda x: pd.Timestamp(x, unit='s'))
    self.df['time_end'] = self.df['time_end'].apply(lambda x: pd.Timestamp(x, unit='s'))

  def getXandY(self, df = None):
    #if df == None:
    #  df = self.df
    return df[self.XCols], df[self.label]

  def filterByDate(self, timeStart, timeEnd, df = None):
    if df is None:
      df = self.df
    mask = (df['time_start'] >= timeStart) & (df['time_start'] < timeEnd)
    
    return df[mask]

  def saveModel(self):
    dump(self.model, os.path.join(self.modelBaseDir, "{}_{}.joblib".format(self.modelType, os.path.basename(self.trainFile))))

  def getModel(self):
    if self.modelType == "rfc":
      return RandomForestClassifier()
    elif self.modelType == "svc":
      return SVC()
    elif self.modelType == "dtc":
      return DecisionTreeClassifier()
    elif self.modelType == "knn":
      return KNeighborsClassifier(n_neighbors=3)
    else:
      print("No model defined")
      return None

if __name__ == "__main__":
  e = Evaluator("rfc") 
  startDate = '2019-04-28'
  endDate = '2019-05-01'
  e.loadData(sys.argv[1])
  e.incrementalTrainEval(startDate, endDate)

  sys.exit()
  if sys.argv[1] == "train":
    e = Evaluator(sys.argv[2])
    e.trainModel(sys.argv[3])
    e.saveModel()
  elif sys.argv[1] == "test":
    e = Evaluator()
    e.loadModel(sys.argv[2])
    e.testModel(sys.argv[3])
    e.evaluateModel()
