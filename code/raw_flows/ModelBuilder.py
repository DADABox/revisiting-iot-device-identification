import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, LSTM, Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras import backend as K

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics

import sys, os, time, json, datetime, glob

from logzero import logger
import warnings
import tensorflow as tf

import numpy as np
import pandas as pd
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.simplefilter("ignore", category=FutureWarning)


class ModelBuilder(object):

  def __init__(self, options):
    self.options = options

  def inferParameters(self):
    if self.options.label == "deviceId":
      self.outputSize = 59
    else:
      self.outputSize = 7

    if self.options.perDeviceModel:
      self.outputSize = 1
      self.activation = "sigmoid"
      self.loss = keras.losses.binary_crossentropy
    else:
      self.activation = "softmax"
      self.loss = keras.losses.categorical_crossentropy

    if self.options.modelType == "nn":
      self.input = 19
    elif self.options.modelType == "lstm":
      self.input = (1, 19)
    elif self.options.modelType == "conv1d":
      self.input = (19, 1)

    self.optimizer = 'adam'
    self.metrics = ['accuracy']

  def getSciKitModel(self):
      return KerasClassifier(build_fn=self.getModel)
  
  def getModel(self):
    self.model = Sequential()
    
    if self.options.modelType == "nn":
      self._getNNModel()
    elif self.options.modelType == "lstm":
      self._getLSTMModel()
    elif self.options.modelType == "conv1d":
      self._getConv1DModel()

    self._addOutputLayer()
    self._compileModel()

    return self.model

  def _addOutputLayer(self):
    self.model.add(Dense(self.outputSize, activation = self.activation))

  def _compileModel(self):
    self.model.compile(loss = self.loss, optimizer = self.optimizer,
            metrics = self.metrics)
  
  def _getNNModel(self):
    # create model
    self.model.add(Dense(32, input_dim = self.input, activation = 'relu'))
    self.model.add(Dense(64, activation = 'relu'))
    self.model.add(Dense(128, activation = 'relu'))
    self.model.add(Dense(256, activation = 'relu'))

  def _getReverseNNModel(self):
    self.model.add(Dense(256, input_dim = self.input, activation = 'relu'))
    self.model.add(Dense(128, activation = 'relu'))
    self.model.add(Dense(64, activation = 'relu'))
    self.model.add(Dense(256, activation = 'relu'))



  def _getLSTMModel(self):
    self.model.add(LSTM(200, activation = 'relu', return_sequences = True, input_shape = self.input))
    self.model.add(LSTM(100, activation = 'relu', return_sequences = True))
    self.model.add(LSTM(50, activation = 'relu', return_sequences = True))
    self.model.add(LSTM(25, activation = 'relu'))
    self.model.add(Dropout(0.2))
  
  def _getConv1DModel(self):
    self.model.add(Conv1D(64, 3, activation = 'relu', input_shape = self.input))
    self.model.add(Conv1D(64, 3, activation = 'relu'))
    self.model.add(Dropout(0.2))
    self.model.add(MaxPooling1D())
    self.model.add(Flatten())
    self.model.add(Dense(100, activation = 'relu'))
    
  def freezeLayers(self, numOfLayers):
    frozenLayers = 0
    for layer in self.model.layers:
      if layer.name.startswith(("dense", "lstm", "conv")):
        layer.trainable = False
        frozenLayers+= 1
      
      if frozenLayers == numOfLayers:
        break

    self._compileModel()
      

if __name__ == '__main__':
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', strides=(2,2), input_shape=(10,250,1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (2, 2), activation='relu', strides=(2, 2)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    #model.add(Bidirectional(LSTM(32, activation = 'relu', return_sequences=True), input_shape=(64,1)))
    #model.add(Bidirectional(LSTM(32, activation = 'relu', return_sequences=False)))
    #model.add(Dropout(0.5))
    model.add(Dense(41, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer='adam',
	metrics=["accuracy"])

    y = []
    packets = [] 
    i = 0
    
    devs = ['appkettle','blink-security-hub','bosiwo-camera-wifi','lefun-cam-wired','insteon-hub',
            'echoplus','meross-dooropener','smartlife-bulb','xiaomi-ricecooker','ubell-doorbell',
            'appletv','tplink-bulb','google-home','icsee-doorbell','t-wemo-plug','echospot',
            'nest-tstat','sousvide','smartlife-remote','netatmo-weather-station','lgtv-wifi',
            'wansview-cam-wired','xiaomi-plug','xiaomi-hub','lightify-hub','bosiwo-camera-wired',
            'tplink-plug2','allure-speaker','honeywell-thermostat','smarter-coffee-mach','roku-tv',
            'yi-camera','firetv','echodot','smartthings-hub','reolink-cam-wired','t-philips-hub',
            'switchbot-hub','ring-doorbell','blink-camera','samsungtv-wired']


    #weeksToProcess = list(range(44,53)) + list(range(1,19))
    weeks = [list(range(44,53)), list(range(1,10)), list(range(10,19))]
    weeksToProcess = weeks[0] + weeks[1] + weeks[2]
    #weeksToProcess = weeks[2]
    #test = False
    test = True

    modelName = "model_mar-apr/model.50-0.07.h5"

    #a = np.fromfile('/data/roman/flows/10_3_2020-03-12_21.06.19_192.168.20.192.pcap.bin', dtype='uint8')
    for week in weeksToProcess:
        print(f"Week {week}")
        y = []
        packets = []
        i = 0
        
        for filename in Path(sys.argv[1]).iterdir():
            devId, numRec, date, time, ip = filename.name.split('_')
            #if date.startswith("2019-12-"): # or date.startswith("2019-12-"):
            if pd.Timestamp(date).week == week: #in weeksToProcess: 
            #if pd.Timestamp(date).week in weeksToProcess: 
                y.extend([int(devId)]*int(numRec))

                #print(devId, numRec, date, time, ip, filename.name)
                x = np.fromfile(filename, dtype='uint8')
                x = np.reshape(x, (int(numRec), 10, 250, 1))
                packets.append(x)

                i+=1

                #if i > 100:
                #    break

        X = np.concatenate(packets)
        print(X.shape)

        X = np.array(X, dtype='float') / 255.0

        lb = LabelBinarizer()
        lb.fit(range(1,42))
        labels = lb.transform(y)
        #labels = keras.utils.to_categorical(y, num_classes = 41)

        if test:
            model = keras.models.load_model(modelName)
            y_predict = model.predict(x=X, batch_size=128)

            print(y_predict.shape, labels.shape, len(lb.classes_))
            
            print(classification_report(labels.argmax(axis=1), y_predict.argmax(axis=1)))#, labels=devs))
            print("Accuracy: ", metrics.accuracy_score(labels.argmax(axis=1), y_predict.argmax(axis=1)), 
                  "F1: ", metrics.f1_score(labels.argmax(axis=1), y_predict.argmax(axis=1)))

        else: # train
            model = keras.models.load_model('model_jan-part-feb/model.50-0.05.h5')
            (trainX, testX, trainY, testY) = train_test_split(X, labels,
                test_size=0.25, stratify=labels, random_state=42)

            my_callbacks = [
                #tf.keras.callbacks.EarlyStopping(patience=2),
                keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1),
                #tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            ]

            H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=50, verbose=1, callbacks=my_callbacks) 
            #H = model.fit(X, labels, epochs=10, verbose=2) 
            break
