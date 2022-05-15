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
import pickle
import time
import numpy as np
from sklearn.metrics import classification_report

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


client_list = []

s = socket.socket()        
print ("Socket successfully created")
port = 12345               
s.bind(('', port))        
print ("socket binded to %s" %(port))

s.listen(5)    
print ("Awaiting all clients handshake...")           

recieveData = False

count = 0

blockHashes = []
fl_round = 1

while count<2:

    c, addr = s.accept()    
    print ('Got connection from', addr )
    client_list.append(c)
    recieveData = True
    count+=1

    while recieveData == True:
        data = c.recv(1024).decode()
        if len(data)>1:
            if data == "Ready":
                print("Client ready, sending update request...")
                c.send('Update'.encode())

            else:
                blockHash = data
                blockHashes.append(blockHash)
                print("Recieved block hash",blockHash)
                c.close()
                recieveData = False
                # break

print(blockHashes)

print("Performing Federated Learning...")

time.sleep(2)

# Get weights from blockchain

with open('blockchain.pickle', 'rb') as handle:
    blockchain = pickle.load(handle)

w1 = blockchain.map[blockHashes[0]].weights
w2 = blockchain.map[blockHashes[1]].weights

w1 = np.array(w1)
w2 = np.array(w2)

# Perform FedAvg algo

w_f = np.mean([w1,w2],axis=0)

# Test model with new weights and print acc

model = define_classifier((28,28,1))
model.set_weights(w_f)



pred = model.predict(x_test)

pred = np.argmax(pred, axis=1)

print(pred)
print(y_test)

print(classification_report(y_test, pred))
# Save weights as weights.npy

# with open('weights.npy','wb') as f:
#     np.save(f,w_f)
with open('weights.pickle', 'wb') as handle:
    pickle.dump(w_f, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Federated Learning Round Finished")


# client_list[0].send("Update".encode())



    #   c.send('Thank you for connecting'.encode())

    #   c.close()
