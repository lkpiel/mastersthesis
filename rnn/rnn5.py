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

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.0001, verbose=1)


model_5 = Sequential([
    Masking(mask_value=0., input_shape=(1107,20)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(3, activation='softmax')
])

print ("model_5 BUILT")

model_5.compile(loss='categorical_crossentropy', optimizer=Adam(0.1), metrics=['accuracy'])
print ("model_5 COMPILED")


checkpoint = ModelCheckpoint(filepath='/home/hpc_lkpiel/models/model_5.hdf5', monitor='val_loss', save_best_only=True)

history = model_5.fit(x=train_data_padded,
        y=train_labels_age_group,
        validation_data=(val_data_padded, val_labels_age_group),
        epochs=25,
        verbose=1,
        batch_size=128,
        callbacks=[checkpoint, reduce_lr]
)

np.save('../history/history_model_5.npy', history.history)
modelHistory = np.load('../history/history_model_5.npy').item()

print ("HISTORY: ")
print (modelHistory)
model_5.load_weights('/home/hpc_lkpiel/models/model_5.hdf5')

valResult = model_5.evaluate(val_data_padded, val_labels_age_group)
testResult = model_5.evaluate(test_data_padded, test_labels_age_group)

file = open("results.txt","a")
file.write("\nmodel_5 VAL: " + str(valResult) + " TEST: " + str(testResult))
file.close()
print ("WROTE TO FILE")


########################################