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

'''
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
test_i_vectors = test_data.iloc[:, 5:].as_matrix()'''

train_data_males = train_data[train_data['Gender'] == 0]
val_data_males = val_data[val_data['Gender'] == 0]
test_data_males = test_data[test_data['Gender'] == 0]
test_data_females = test_data[test_data['Gender'] == 1]


train_labels_males =  onehot(train_data_males['AgeGroup'])
val_labels_males = onehot(val_data_males['AgeGroup'])
test_labels_males = onehot(test_data_males['AgeGroup'])
test_labels_females = onehot(test_data_females['AgeGroup'])



train_i_vectors_males = train_data_males.iloc[:, 5:].as_matrix()
val_i_vectors_males = val_data_males.iloc[:, 5:].as_matrix()
test_i_vectors_males = test_data_males.iloc[:, 5:].as_matrix()
test_i_vectors_females = test_data_females.iloc[:, 5:].as_matrix()


print ("DATA FORMATTED")


################################################################################################




reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                              patience=2, min_lr=0.0001, verbose=1)

dnn_1 = Sequential([
    Dense(2000, activation='relu', input_shape=(600,)),
    Dropout(0.6),
    Dense(2000, activation='relu'),
    Dropout(0.2),
    Dense(2000, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')
])

print ("dnn_1 MODEL BUILT")

dnn_1.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'])
print ("dnn_1 MODEL COMPILED")


checkpoint = ModelCheckpoint(filepath='/home/hpc_lkpiel/models/dnn_1.hdf5', monitor='val_acc', save_best_only=True)

'''
history = dnn_1.fit(
    x=train_i_vectors,
    y=train_labels_age_group,
    validation_data=(val_i_vectors, val_labels_age_group),
    epochs=20,
    batch_size=16,
    callbacks=[reduce_lr, checkpoint]

)
np.save('../history/dnn/dnn_1.npy', history.history)
modelHistory = np.load('../history/dnn/dnn_1.npy').item()

print ("HISTORY: ")
print (modelHistory)

dnn_1.load_weights('/home/hpc_lkpiel/models/dnn_1.hdf5')

testResult = dnn_1.evaluate(test_i_vectors, test_labels_age_group)

print (testResult)
print ("DONE dnn_1")




'''
dnn_1.load_weights('/home/hpc_lkpiel/models/dnn_1.hdf5')

val_predictions = dnn_1.predict(val_i_vectors_males)
print ("VAL PREDICTED")

test_predictions = dnn_1.predict(test_i_vectors_males)
print ("TEST PREDICTED")

np.save('/home/hpc_lkpiel/predictions/val/dnn_1_males_original.npy', val_predictions)
print ("VAL SAVED")

np.save('/home/hpc_lkpiel/predictions/test/dnn_1_males_original.npy', test_predictions)

print ("TEST SAVED")
########################################