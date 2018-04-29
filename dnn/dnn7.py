#! /usr/bin/python2.7

import sys
print(sys.version)
import sys
import pandas
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import Input
from keras.optimizers import SGD, Adam
from IPython.core.debugger import Tracer
from keras.layers import Masking, LSTM, TimeDistributed, Bidirectional, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model



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


train_data_males = train_data[train_data['Gender'] == 0]
train_data_females = train_data[train_data['Gender'] == 1]
val_data_males = val_data[val_data['Gender'] == 0]
val_data_females = val_data[val_data['Gender'] == 1]


train_labels_males =  onehot(train_data_males['AgeGroup'])
val_labels_males = onehot(val_data_males['AgeGroup'])

train_i_vectors_males = train_data_males.iloc[:, 5:].as_matrix()
val_i_vectors_males = val_data_males.iloc[:, 5:].as_matrix()

train_labels_females =  onehot(train_data_females['AgeGroup'])
val_labels_females = onehot(val_data_females['AgeGroup'])

train_i_vectors_females = train_data_females.iloc[:, 5:].as_matrix()
val_i_vectors_females = val_data_females.iloc[:, 5:].as_matrix()
print ("DATA FORMATTED")


################################################################################################


import tensorflow as tf
import keras
from keras import backend as K
def age_group_accuracy(y_true, y_pred):
        array = np.array([0]*13 + [1]*2 + [2]*13000000)

        age_to_group = K.variable(value=array, dtype='int32', name='age_to_group')
        ages_true = tf.gather(age_to_group, tf.cast(tf.rint(y_true), tf.int32))
        ages_pred = tf.gather(age_to_group, tf.cast(tf.rint(y_pred), tf.int32))
        return K.mean(K.equal(ages_true, ages_pred), axis=-1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              patience=3, min_lr=0.0005, verbose=1)


input_layer = Input(shape=(600,))
x = Dense(3000, activation='relu')(input_layer)
x = Dropout(0.6)(x)
x = Dense(1500, activation='relu')(input_layer)
x = Dropout(0.4)(x)

output_layer_1 = Dense(2, activation='softmax', name='gender_output')(x)

y = Dense(100, activation='sigmoid')(x)
output_layer_2 =  Dense(1, name='age_output')(y)



multi_model = Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])

multi_model.compile(loss={'gender_output':'categorical_crossentropy', 'age_output':'mse'},
                    optimizer=SGD(0.01),
                    metrics={'gender_output':'accuracy','age_output':age_group_accuracy})
print ("dnn_7 MODEL COMPILED")


checkpoint = ModelCheckpoint(filepath='/home/hpc_lkpiel/models/dnn_7.hdf5', monitor='val_age_output_age_group_accuracy', save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_group_output_loss', factor=0.8,
                              patience=1, min_lr=0.0001, verbose=1)

history = multi_model.fit(
    train_i_vectors,
    [train_labels_gender, train_labels_age],
    validation_data=(val_i_vectors, [val_labels_gender, val_labels_age]),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint]
)
np.save('../history/dnn/dnn_7.npy', history.history)
modelHistory = np.load('../history/dnn/dnn_7.npy').item()

print ("HISTORY: ")
print (modelHistory)
print ("DONE dnn_7")


multi_model.load_weights('/home/hpc_lkpiel/models/dnn_7.hdf5')


val_predictions = multi_model.predict(val_i_vectors)
print ("VAL PREDICTED")

test_predictions = multi_model.predict(test_i_vectors)
print ("TEST PREDICTED")


np.save('/home/hpc_lkpiel/predictions/val/dnn_7_gender.npy', val_predictions[0])
print ("VAL SAVED")

np.save('/home/hpc_lkpiel/predictions/test/dnn_7_gender.npy', test_predictions[0])
print ("TEST SAVED")


np.save('/home/hpc_lkpiel/predictions/val/dnn_7_age_group.npy', val_predictions[1])
print ("VAL SAVED")

np.save('/home/hpc_lkpiel/predictions/test/dnn_7_age_group.npy', test_predictions[1])
print ("TEST SAVED")

multi_model.evaluate(val_i_vectors)

########################################