import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.initializers import RandomNormal
from keras.layers import Activation, Dense, Dropout, Flatten, ReLU
from keras.models import Model
from keras.models import Input
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
from sklearn.model_selection import train_test_split

import socket
import os
import pickle
import json

from blockchain import *

def define_classifier(image_shape):

  init = RandomNormal(stddev=0.02)
  in_image = Input(shape=image_shape)

  c = Flatten()(in_image)

  c = Dense(16, activation='relu', kernel_initializer=init)(c)
  c = Dropout(0.1)(c)

  c = Dense(32, activation='relu', kernel_initializer=init)(c)
  c = Dropout(0.1)(c)

  c = Dense(64, activation='relu', kernel_initializer=init)(c)
  c = Dropout(0.1)(c)

  c = Dense(128, activation='relu', kernel_initializer=init)(c)
  c = Dropout(0.2)(c)

  c = Dense(256, activation='relu', kernel_initializer=init)(c)
  c = Dropout(0.3)(c)

  out = Dense(10, activation = 'softmax', kernel_initializer=init)(c)

  model = Model(in_image, out)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
  return model

with open('x_data.npy','rb') as f:
  x1 = np.load(f)
  x2 = np.load(f)
  x3 = np.load(f)
  x_test = np.load(f)

with open('y_data.npy','rb') as f:
  y1 = np.load(f)
  y2 = np.load(f)
  y3 = np.load(f)
  y_test = np.load(f)


def create_sub_samples(x,y):
  X_sample1 = x[:1000]
  # X_sample2 = x[1000:2000]
  # X_sample3 = x[2000:]

  # y_sample1 = y[:1000]
  # y_sample2 = y[1000:2000]
  # y_sample3 = y[2000:]
  X_sample1 = x[:5000]
  X_sample2 = x[5000:10000]
  X_sample3 = x[10000:]

  y_sample1 = y[:5000]
  y_sample2 = y[5000:10000]
  y_sample3 = y[10000:]

  return [X_sample1, X_sample2, X_sample3], [y_sample1, y_sample2, y_sample3]


X_sample_list, y_sample_list = create_sub_samples(x1,y1)


def trainRound(x,y):

  x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33, random_state=42)

  x_train_reg = x_train.astype('float32') / 255.0
  y_train_cat =  to_categorical(y_train)


  if os.path.exists('weights.pickle') == True:
    model = define_classifier((28,28,1))
    with open('weights.pickle','rb') as f:
      weights = pickle.load(f)

    model.set_weights(weights)

  else:
    model = define_classifier((28,28,1))

  model.fit(x_train_reg, y_train_cat, epochs=10, batch_size=64)

  return model

def performFederated(round_number):
  x = X_sample_list[round_number]
  y = y_sample_list[round_number]

  model = trainRound(x,y)

  weights = model.get_weights()

  return weights


def main():
  round_number = 0
  file_path = 'meta_data_client_1.json'
  if os.path.exists(file_path) == True:
    with open(file_path,'r') as f:
      md = json.load(f)
      round_number = md['round_number']

  print("Federated Learning Round:",round_number)

  clientID = "1"

  with open('blockchain.pickle', 'rb') as handle:
      blockchain = pickle.load(handle)

  s = socket.socket()        
  port = 12345               
  s.connect(('127.0.0.1', port))
  s.send("Ready".encode())
  while True:
    data = s.recv(1024).decode()
    if len(data)>1:
      print(data)
      print(len(blockchain.map.keys()))
      if data == "Update":
        print("Training....")
        weights = performFederated(round_number)
        print("Adding block to chain...")
        blockHash = blockchain.make_block(clientID,weights)
        print('Sending block hash')
        s.send(blockHash.encode())

        print(len(blockchain.map.keys()))
        with open('blockchain.pickle', 'wb') as handle:
            pickle.dump(blockchain, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save meta data of round number and load it in runs

        meta_data_client_1 = {}

        meta_data_client_1['round_number'] = round_number+1

        with open(file_path,'w') as f:
          json.dump(meta_data_client_1,f,indent=4)

        break



  # # close the connection
  # s.close()    
     

    # listen to request from server
    # send to server
    # wait for server response
    # set weights
    # weights = server_weights

main()









