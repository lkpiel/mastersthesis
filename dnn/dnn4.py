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
print ("DATA FORMATTED")


################################################################################################




reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                              patience=2, min_lr=0.0001, verbose=1)
from keras import initializers
input_layer = Input(shape=(600,))
x = Dense(3000, activation='relu')(input_layer)
x = Dropout(0.7)(x)
x = Dense(1500, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(300, activation='relu')(x)
x = Dropout(0.3)(x)

output_layer_1 = Dense(3, activation='softmax', name='group_output')(x)
output_layer_2 =  Dense(2, activation='softmax', name='gender_output')(x)

print ("dnn_4 MODEL BUILT")

multi_model = Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])

multi_model.compile(loss={'group_output':'categorical_crossentropy', 'gender_output':'categorical_crossentropy'},
                    optimizer='sgd',
                    metrics={'group_output':'accuracy','gender_output':'accuracy'})

print ("dnn_4 MODEL COMPILED")


checkpoint = ModelCheckpoint(filepath='/models/dnn_4_gender.hdf5', monitor='val_gender_output_acc', save_best_only=True)

history = multi_model.fit(
    x=train_i_vectors,
    y=[train_labels_age_group, train_labels_gender],
    validation_data=(val_i_vectors, [val_labels_age_group, val_labels_gender]),
    epochs=100,
    batch_size=100,
    callbacks=[checkpoint]
)

np.save('../history/dnn/dnn_4_gender.npy', history.history)
modelHistory = np.load('../history/dnn/dnn_4_gender.npy').item()
multi_model.load_weights('/models/dnn_4_gender.hdf5')
print ("HISTORY: ")
print (modelHistory)
print ("DONE dnn_4")



print (str(multi_model.evaluate(val_i_vectors, [val_labels_age_group, val_labels_gender])))

val_predictions = multi_model.predict(val_i_vectors)
print ("VAL PREDICTED")

test_predictions = multi_model.predict(test_i_vectors)
print ("TEST PREDICTED")




np.save('/home/hpc_lkpiel/predictions/val/dnn_4_gender.npy', val_predictions[1])

print ("VAL SAVED")

np.save('/home/hpc_lkpiel/predictions/test/dnn_4_gender.npy', test_predictions[1])

print ("TEST SAVED")

########################################