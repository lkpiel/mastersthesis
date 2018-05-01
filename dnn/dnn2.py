#! /usr/bin/python2.7

import sys
print(sys.version)
import sys
import pandas
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, Adam
from IPython.core.debugger import Tracer
from keras.layers import Masking, LSTM, TimeDistributed, Bidirectional, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder



#LOAD DATA
train_data = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/train/export.csv", sep=" ")
val_data = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/dev/export.csv", sep=" ")
test_data = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/test/export.csv", sep=" ")
print ("DATA LOADED")


#FORMAT DATA
#ONE HOT ENCODES A GIVEN COLUMN
def onehot(x): return np.array(OneHotEncoder().fit_transform(x.values.reshape(-1,1)).todense())

def format(data):
    del data['Unnamed: 605']
    mask = data['AgeGroup'] == 'ag1'
    column_name = 'AgeGroup'
    data.loc[mask, column_name] = 0
    mask = data['AgeGroup'] == 'ag2'
    column_name = 'AgeGroup'
    data.loc[mask, column_name] = 1
    mask = data['AgeGroup'] == 'ag3'
    column_name = 'AgeGroup'
    data.loc[mask, column_name] = 2
    mask = data['Gender'] == 'm'
    column_name = 'Gender'
    data.loc[mask, column_name] = 0
    mask = data['Gender'] == 'f'
    column_name = 'Gender'
    data.loc[mask, column_name] = 1
    return data

train_data = format(train_data)
val_data = format(val_data)
test_data = format(test_data)


train_labels_age_group = onehot(train_data['AgeGroup'])
val_labels_age_group = onehot(val_data['AgeGroup'])
test_labels_age_group = onehot(test_data['AgeGroup'])

train_labels_age = train_data['Age']
val_labels_age = val_data['Age']
test_labels_age = test_data['Age']

train_labels_gender =  onehot(train_data['Gender'])
val_labels_gender = onehot(val_data['Gender'])
test_labels_gender = onehot(test_data['Gender'])

train_i_vectors = train_data.iloc[:, 5:].as_matrix()
val_i_vectors = val_data.iloc[:, 5:].as_matrix()
test_i_vectors = test_data.iloc[:, 5:].as_matrix()

print ("DATA FORMATTED")


################################################################################################


import tensorflow as tf
import keras
from keras import backend as K
def age_group_accuracy(y_true, y_pred):
    y_pred = tf.nn.relu(y_pred)

    array = np.array([0]*13 + [1]*2 + [2]*13000000)
    age_to_group = K.variable(value=array, dtype='int32', name='age_to_group')
    ages_true = tf.gather(age_to_group, tf.cast(tf.rint(y_true), tf.int32))
    ages_pred = tf.gather(age_to_group, tf.cast(tf.rint(y_pred), tf.int32))
    return K.mean(K.equal(ages_true, ages_pred), axis=-1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                              patience=2, min_lr=0.0001, verbose=1)

dnn_2 = Sequential([
    Dense(2000, activation = 'relu', input_shape=(600,)),
    Dropout(0.7),
    Dense(1000, activation = 'relu'),
    Dropout(0.4),
    Dense(100, activation = 'sigmoid'),
    Dense(1)
])

print ("dnn_2 MODEL BUILT")

dnn_2.compile(
    loss='mse',
    optimizer=SGD(lr=0.01),
    metrics=[age_group_accuracy]
)
print ("dnn_2 MODEL COMPILED")


checkpoint = ModelCheckpoint(filepath='/models/dnn_2.hdf5', monitor='val_age_group_accuracy', save_best_only=True)

history = dnn_2.fit(
    x=train_i_vectors,
    y=train_labels_age,
    validation_data=(val_i_vectors, val_labels_age),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint]
)

np.save('../history/dnn/dnn_2.npy', history.history)

dnn_2.load_weights('/models/dnn_2.hdf5')



print ("EVALUATION ", str(dnn_2.evaluate(val_i_vectors, val_labels_age)))




valPredictions = dnn_2.predict(val_i_vectors)
testPredictions = dnn_2.predict(test_i_vectors)


np.save("../predictions/val/dnn_2.npy", valPredictions)
np.save("../predictions/test/dnn2.npy", testPredictions)

#print ("HISTORY: ")
#print (modelHistory)
print ("DONE dnn_2")


########################################