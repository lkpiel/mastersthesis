#! /usr/bin/python3


import sys
print(sys.version)
import sys
import pandas
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Average, Merge, Layer, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling2D, AveragePooling2D, Reshape, BatchNormalization
from keras.optimizers import SGD, Adam
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from IPython.core.debugger import Tracer
from keras.layers import Masking, LSTM, TimeDistributed, Bidirectional, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import callbacks
from keras.constraints import maxnorm, unitnorm
from sklearn.preprocessing import OneHotEncoder

import pdb
import keras

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



#LOAD LABELS
train_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/train/export.csv", sep=" ")
train_data_i_vectors = format(train_data_i_vectors)
train_labels_age_group = onehot(train_data_i_vectors['AgeGroup'])

val_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/dev/export.csv", sep=" ")
val_data_i_vectors = format(val_data_i_vectors)
val_labels_age_group = onehot(val_data_i_vectors['AgeGroup'])

test_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/test/export.csv", sep=" ")
test_data_i_vectors = format(test_data_i_vectors)
test_labels_age_group = onehot(test_data_i_vectors['AgeGroup'])
print ("LABELS LOADED")


#LOAD DATA

train_data_padded = np.load("/storage/hpc_lkpiel/data/fbank_train_data_padded.npy", encoding="bytes")
val_data_padded = np.load("/storage/hpc_lkpiel/data/fbank_val_data_padded.npy", encoding="bytes")
test_data_padded = np.load("/storage/hpc_lkpiel/data/fbank_test_data_padded.npy", encoding="bytes")
print ("DATA LOADED")

################################################################################################

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                              patience=2, min_lr=0.0001, verbose=1)


model_13 = Sequential([
    Masking(mask_value=0., input_shape=(1107,20)),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.4),
    Bidirectional(LSTM(128)),
    Dense(3, activation='softmax')
])

print ("model_13 BUILT")

model_13.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print ("model_13 COMPILED")


checkpoint = ModelCheckpoint(filepath='/models/model_13.hdf5', monitor='val_loss', save_best_only=True)

history = model_13.fit(x=train_data_padded,
        y=train_labels_age_group,
        validation_data=(val_data_padded, val_labels_age_group),
        epochs=25,
        verbose=1,
        batch_size=128,
        callbacks=[checkpoint]
)

np.save('../history/history_model_13.npy', history.history)
modelHistory = np.load('../history/history_model_13.npy').item()

print ("HISTORY: ")
print (modelHistory)
model_13.load_weights('/models/model_13.hdf5')

valResult = model_13.evaluate(val_data_padded, val_labels_age_group)
testResult = model_13.evaluate(test_data_padded, test_labels_age_group)

file = open("results.txt","a")
file.write("\nmodel_13 VAL: " + str(valResult) + " TEST: " + str(testResult))
file.close()
print ("WROTE TO FILE")


########################################