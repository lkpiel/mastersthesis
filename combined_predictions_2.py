#! /usr/bin/python2.7


import sys
print(sys.version)
import sys
import pandas
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Dropout, Average, Merge, Layer, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling2D, AveragePooling2D, Reshape, BatchNormalization
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
from keras.models import Model
import pdb
import keras
from collections import Counter


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

train_i_vectors = train_data.iloc[:, 5:].as_matrix()
val_i_vectors = val_data.iloc[:, 5:].as_matrix()
test_i_vectors = test_data.iloc[:, 5:].as_matrix()

val_labels_age_group_females = onehot(val_data['AgeGroup'][val_data['Gender'] == 1])
test_labels_age_group_females = onehot(test_data['AgeGroup'][test_data['Gender'] == 1])
val_labels_age_group = onehot(val_data['AgeGroup'])

test_labels_age_group = onehot(test_data['AgeGroup'])


val_data_padded = np.load("/storage/hpc_lkpiel/data/fbank_val_data_padded.npy", encoding="bytes")[..., np.newaxis]
val_data_padded_no_extra_dim = np.load("/storage/hpc_lkpiel/data/fbank_val_data_padded.npy", encoding="bytes")
test_data_padded = np.load("/storage/hpc_lkpiel/data/fbank_test_data_padded.npy", encoding="bytes")[..., np.newaxis]
test_data_padded_no_extra_dim = np.load("/storage/hpc_lkpiel/data/fbank_test_data_padded.npy", encoding="bytes")

print ("DATA FORMATTED")



#LOAD FEMALES' MODEL ACCURACIES
def getAgeGroupModelAccuracy(predictions, data):
    correct_predictions = []
    for i in range(0, len(predictions)):
        if (np.argmax(predictions[i]) == np.argmax(data[i])):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    return correct_predictions.count(1)/len(predictions)

model_63_females_original_val = np.load("./val/model_63_females_original.npy", encoding="bytes")
model_63_females_original_test = np.load("./test/model_63_females_original.npy", encoding="bytes")
dnn_6_females_original_val_age = np.load("./val/dnn_6_age_females_original.npy", encoding="bytes")
dnn_6_females_original_test_age = np.load("./test/dnn_6_age_females_original.npy", encoding="bytes")
model_94_females_original_val_age_group = np.load("./val/model_94_females_original_age_group.npy", encoding="bytes")
model_94_females_original_test_age_group = np.load("./test/model_94_females_original_age_group.npy", encoding="bytes")
model_65_females_original_val = np.load("./val/model_65_females_original.npy", encoding="bytes")
model_65_females_original_test = np.load("./test/model_65_females_original.npy", encoding="bytes")

model_63_females_original_val_acc = getAgeGroupModelAccuracy(model_63_females_original_val, val_labels_age_group_females)
dnn_6_females_original_val_age_acc = getAgeGroupModelAccuracy(dnn_6_females_original_val_age, val_labels_age_group_females)
model_94_females_original_val_age_group_acc = getAgeGroupModelAccuracy(model_94_females_original_val_age_group, val_labels_age_group_females)
model_65_females_original_val_acc = getAgeGroupModelAccuracy(model_65_females_original_val, val_labels_age_group_females)

model_63_females_original_test_acc = getAgeGroupModelAccuracy(model_63_females_original_test, test_labels_age_group_females)
dnn_6_females_original_test_age_acc = getAgeGroupModelAccuracy(dnn_6_females_original_test_age, test_labels_age_group_females)
model_94_females_original_test_age_group_acc = getAgeGroupModelAccuracy(model_94_females_original_test_age_group, test_labels_age_group_females)
model_65_females_original_test_acc = getAgeGroupModelAccuracy(model_65_females_original_test, test_labels_age_group_females)


print (model_63_females_original_val_acc)
def smooth_labels(y, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.
    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)
    # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

import tensorflow as tf
import keras
from keras import backend as K

def age_group_accuracy(y_true, y_pred):
  array = np.array([0]*13 + [1]*2 + [2]*10000000)
  age_to_group = K.variable(value=array, dtype='int32', name='age_to_group')
  ages_true = tf.gather(age_to_group, tf.cast(tf.rint(y_true), tf.int32))
  ages_pred = tf.gather(age_to_group, tf.cast(tf.rint(y_pred), tf.int32))
  return K.mean(K.equal(ages_true, ages_pred), axis=-1)


kernel_regularizer = regularizers.l2(0.0001)

################################################################################################



input_layer = Input(shape=(1107, 20))
x =  Masking(mask_value=0.)(input_layer)
x =  Bidirectional(LSTM(64, return_sequences=True))(x)
x =  Bidirectional(LSTM(64, return_sequences=True))(x)
x = AttentionWithContext()(x)
output_layer_1 = Dense(2, activation='softmax', name='gender_output')(x)
y = Dense(100, activation='sigmoid')(x)
output_layer_2 =  Dense(1, name='age_output')(y)
model_94_females_original = Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])
model_94_females_original.compile(loss={'gender_output':'categorical_crossentropy', 'age_output':'mse'},
                    optimizer=SGD(0.01),
                    metrics={'gender_output':'accuracy','age_output':age_group_accuracy})
model_94_females_original.load_weights('/models/model_94.hdf5')





#LOAD GENDER MODEL

dnn_3 = Sequential([
    Dense(4000, activation='relu', input_shape=(600,)),
    Dropout(0.5),
    Dense(2000, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])
dnn_3.compile(loss='categorical_crossentropy', optimizer=SGD(0.001), metrics=['accuracy'])
dnn_3.load_weights('/models/dnn_3.hdf5')
print ("dnn_3 WEIGHTS LOADED")






#LOAD MALES' AGE GROUP MODELS

dnn_2_males = Sequential([
    Dense(2000, activation = 'relu', input_shape=(600,)),
    Dropout(0.7),
    Dense(1000, activation = 'relu'),
    Dropout(0.4),
    Dense(100, activation = 'sigmoid'),
    Dense(1)
])
dnn_2_males.compile(loss='mse',optimizer=SGD(lr=0.01))
dnn_2_males.load_weights('/models/dnn_2_males.hdf5')
print ("dnn_2_males WEIGHTS LOADED")




model_43_males = Sequential([
    Conv2D(128, (3, 20), activation='relu', kernel_regularizer=kernel_regularizer,  border_mode='valid', input_shape=(1107, 20, 1)),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu', kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu',  kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu',  kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu',  kernel_regularizer=kernel_regularizer, border_mode='valid'),

    Reshape((-1, 128)),
    Bidirectional(LSTM(128, return_sequences=True)),
    AttentionWithContext(),
    Dense(3, activation='softmax')
])
model_43_males.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'])
model_43_males.load_weights('/models/model_43_males.hdf5')
print ("model_43 WEIGHTS LOADED")



input_layer = Input(shape=(1107, 20, 1), name="lstm_input")
x = Conv2D(128, (3, 20), activation='relu', kernel_regularizer=kernel_regularizer,  border_mode='valid')(input_layer)
x = Conv2D(128, (5, 1), strides=(3,1), activation='relu', kernel_regularizer=kernel_regularizer, border_mode='valid')(x)
x = Conv2D(128, (5, 1), strides=(3,1), activation='relu', kernel_regularizer=kernel_regularizer, border_mode='valid')(x)
x = Reshape((-1, 128))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = AttentionWithContext()(x)
input_layer_2 = Input(shape=(600,), name = "dnn_input")
y = Dense(4000, activation='relu')(input_layer_2)
y = Dropout(0.7)(y)
z = keras.layers.concatenate([y, x])
z = Dense(1500, activation='relu')(z)
z = Dropout(0.5)(z)
z = Dense(300, activation='relu')(z)
z = Dropout(0.3)(z)
z = Dense(300, activation='relu')(z)
z = Dropout(0.3)(z)
output_layer_1 = Dense(3, activation='softmax')(z)
model_57_males = Model(inputs=[input_layer, input_layer_2], outputs=output_layer_1)
model_57_males.compile(loss='categorical_crossentropy',
                    optimizer=SGD(0.01),
                    metrics=['accuracy'])
model_57_males.load_weights('/models/model_57_males.hdf5')

print ("model_57 WEIGHTS LOADED")




model_65_males = Sequential([
    Conv2D(128, (3, 20), activation='relu', kernel_regularizer=kernel_regularizer,  border_mode='valid', input_shape=(1107, 20, 1)),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu', kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu',  kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu',  kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu',  kernel_regularizer=kernel_regularizer, border_mode='valid'),

    Reshape((-1, 128)),
    Bidirectional(LSTM(128, return_sequences=True)),
    AttentionWithContext(),
    Dense(100, activation="relu"),
    Dropout(0.1),
    Dense(3, activation='softmax')
])
model_65_males.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model_65_males.load_weights('/models/model_65_males.hdf5')
print ("model_65_males WEIGHTS LOADED")










# LOAD FEMALES' AGE GROUP MODELS

input_layer = Input(shape=(600,))
x = Dense(2000, activation='relu')(input_layer)
x = Dropout(0.7)(x)
x = Dense(600, activation='relu')(input_layer)
x = Dropout(0.4)(x)
z = Dense(2000, activation='relu')(x)
z = Dropout(0.4)(z)
output_layer_1 = Dense(3, activation='softmax', name='group_output')(z)
y = Dense(100, activation = 'sigmoid')(x)
output_layer_2 =  Dense(1, name='age_output')(y)
dnn_6_females_original = Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])

dnn_6_females_original.compile(loss={'group_output':'categorical_crossentropy', 'age_output':'mse'},
                    optimizer="sgd",
                    metrics={'group_output':'accuracy','age_output':age_group_accuracy})
dnn_6_females_original.load_weights("/models/dnn_6.hdf5")
print ("dnn_6 MODEL COMPILED")




input_layer = Input(shape=(1107, 20, 1), name="lstm_input")
x = Conv2D(128, (3, 20), activation='relu', kernel_regularizer=kernel_regularizer,  border_mode='valid')(input_layer)
x = Conv2D(128, (5, 1), strides=(3,1), activation='relu', kernel_regularizer=kernel_regularizer, border_mode='valid')(x)
x = Conv2D(128, (5, 1), strides=(3,1), activation='relu', kernel_regularizer=kernel_regularizer, border_mode='valid')(x)
x = Conv2D(128, (5, 1), strides=(3,1), activation='relu', kernel_regularizer=kernel_regularizer, border_mode='valid')(x)
x = Conv2D(128, (5, 1), strides=(3,1), activation='relu', kernel_regularizer=kernel_regularizer, border_mode='valid')(x)
x = Reshape((-1, 128))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = AttentionWithContext()(x)
x = Dense(100)(x)
x = Dropout(0.1)(x)
input_layer_2 = Input(shape=(600,), name = "dnn_input")
z = keras.layers.concatenate([input_layer_2, x])
z = Dense(100)(z)
z = Dropout(0.1)(z)
output_layer_1 = Dense(3, activation='softmax')(z)
model_63_females_original = Model(inputs=[input_layer, input_layer_2], outputs=output_layer_1)
model_63_females_original.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model_63_females_original.load_weights('/models/model_63.hdf5')
print("MODEL 63 females original weights loaded")






model_65_females_original = Sequential([
    Conv2D(128, (3, 20), activation='relu', kernel_regularizer=kernel_regularizer,  border_mode='valid', input_shape=(1107, 20, 1)),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu', kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu',  kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu',  kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Conv2D(128, (5, 1), strides=(3,1), activation='relu',  kernel_regularizer=kernel_regularizer, border_mode='valid'),
    Reshape((-1, 128)),
    Bidirectional(LSTM(128, return_sequences=True)),
    AttentionWithContext(),
    Dense(100, activation="relu"),
    Dropout(0.1),
    Dense(3, activation='softmax')
])
model_65_females_original.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model_65_females_original.load_weights('/models/model_65.hdf5')






def formatAgePredictions(data):
    predictions = []
    for i in range(0, len(data)):
        if (np.round(data[i]) < 13):
            predictions.append([1, 0, 0])
        elif (np.round(data[i]) < 15):
            predictions.append([0, 1, 0])
        else:
            predictions.append([0, 0, 1])
    return predictions




def averagePredictions(predictions, weights, data):
    correct_predictions = []

    for i in range(0, len(predictions[0])):
        predicted = np.zeros(len(predictions[0][0]))

        for j in range(0, len(predictions)):
            predicted += (predictions[j][i]*(weights[j]))

        if (np.argmax(predicted) == np.argmax(data[i])):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    return correct_predictions.count(1)/len(correct_predictions)


def majorityVoting(predictions, accuracies):
    correct_predictions = []
    predicted = []
    for i in range(0, len(predictions)):
        predicted.append(np.argmax(predictions[i]))
    c = Counter(predicted)
    commonElements = c.most_common(3)
    # CHECK IF ONE ELEMENT WAS MORE POPULAR THAN OTHERS
    prediction = -1
    if (len(commonElements) > 1):
        if (commonElements[0][1] == commonElements[1][1]):
            prediction = np.argmax(predictions[np.argmax(accuracies)])
        else:
            prediction = commonElements[0][0]
    else:
        prediction = commonElements[0][0]

    if (prediction == 0):
        return [0.5, 0.0, 0.0]
    elif prediction == 1:
         return [0.0, 0.5, 0.0]
    elif prediction == 2:
        return [0.0, 0.0, 0.5]




combined_predictions_val = []
combined_predictions_test = []


def combinePredictions(i_vectors, sequence_data, predictions, accuracies):
    combined_predictions = []
    for i in range(0, len(i_vectors)):
        vector =  np.expand_dims(i_vectors[i], axis=0)
        padded_data = np.expand_dims(sequence_data[i], axis=0)
        gender_prediction = dnn_3.predict(vector)

        if (np.argmax(gender_prediction) == 1):
            age_group_prediction = majorityVoting(
                [
                    formatAgePredictions(predictions[0][i]),
                    predictions[1][i],
                    formatAgePredictions(predictions[2][i]),
                    predictions[3][i]
                ],
                accuracies
            )
        else:
            age_group_prediction =  0.3 * np.squeeze(model_43_males.predict(padded_data)) + 0.3 * np.squeeze(model_57_males.predict([padded_data, vector])) + 0.3 * np.squeeze(formatAgePredictions(dnn_2_males.predict(vector))) + 0.2 * np.squeeze(model_65_males.predict(padded_data))
        age_group_prediction = age_group_prediction
        combined_predictions.append(age_group_prediction)

    return combined_predictions



def combineWeightedPredictions(i_vectors, sequence_data, predictions, accuracies):
    combined_predictions = []
    for i in range(0, len(i_vectors)):
        vector =  np.expand_dims(i_vectors[i], axis=0)
        padded_data = np.expand_dims(sequence_data[i], axis=0)
        gender_prediction = dnn_3.predict(vector)
        age_group_prediction_female_model = majorityVoting(
            [
                formatAgePredictions(predictions[0][i]),
                predictions[1][i],
                formatAgePredictions(predictions[2][i]),
                predictions[3][i]
            ],
            accuracies
        )
        print ("FEMALE ", str(age_group_prediction_female_model))
        age_group_prediction_male_model =  0.3 * np.squeeze(model_43_males.predict(padded_data)) + 0.3 * np.squeeze(model_57_males.predict([padded_data, vector])) + 0.3 * np.squeeze(formatAgePredictions(dnn_2_males.predict(vector))) + 0.2 * np.squeeze(model_65_males.predict(padded_data))
        print ("MALE ", str(age_group_prediction_male_model))
        print (gender_prediction)

        age_group_prediction = np.expand_dims(gender_prediction[0][1], axis=0) * age_group_prediction_female_model + np.expand_dims(gender_prediction[0][0], axis=0) * age_group_prediction_male_model
        print ("COMBINED ", str(age_group_prediction))
        combined_predictions.append(age_group_prediction)

    return combined_predictions



def getAgeGroupModelAccuracy(predictions, data):
    correct_predictions = []
    for i in range(0, len(predictions)):
        if (np.argmax(predictions[i]) == np.argmax(data[i])):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    return correct_predictions.count(1)/len(predictions)


dnn_6_females_predictions_val = dnn_6_females_original.predict(val_i_vectors)

print (getAgeGroupModelAccuracy(formatAgePredictions(dnn_6_females_predictions_val[1]) ,val_labels_age_group))
model_63_females_original_predictions_val = model_63_females_original.predict([val_data_padded, val_i_vectors])
print (getAgeGroupModelAccuracy(model_63_females_original_predictions_val, val_labels_age_group))

model_94_females_original_predictions_val = model_94_females_original.predict(val_data_padded_no_extra_dim)
print (getAgeGroupModelAccuracy(formatAgePredictions(model_94_females_original_predictions_val[1]), val_labels_age_group))

model_65_females_original_predictions_val = model_65_females_original.predict(val_data_padded)
print (getAgeGroupModelAccuracy(model_65_females_original_predictions_val, val_labels_age_group))
'''
dnn_6_females_predictions_test = dnn_6_females_original.predict(test_i_vectors)
model_63_females_original_predictions_test = model_63_females_original.predict([test_data_padded, test_i_vectors])
model_94_females_original_predictions_test = model_94_females_original.predict(test_data_padded_no_extra_dim)
model_65_females_original_predictions_test = model_65_females_original.predict(test_data_padded)

'''
'''
print ("STARTING TO COMBINE ON VALIDATION DATA")
combined_predictions_val = combinePredictions(val_i_vectors, val_data_padded,
                [
                    dnn_6_females_predictions_val[1],
                    model_63_females_original_predictions_val,
                    model_94_females_original_predictions_val[1],
                    model_65_females_original_predictions_val
                ],
                [
                    dnn_6_females_original_val_age_acc,
                    model_63_females_original_val_acc,
                    model_94_females_original_val_age_group_acc,
                    model_65_females_original_val_acc
                ])



print ("STARTING TO COMBINE ON TEST DATA")
combined_predictions_test = combinePredictions(test_i_vectors, test_data_padded,
                 [
                    dnn_6_females_predictions_test[1],
                    model_63_females_original_predictions_test,
                    model_94_females_original_predictions_test[1],
                    model_65_females_original_predictions_test
                ],
                [
                    dnn_6_females_original_test_age_acc,
                    model_63_females_original_test_acc,
                    model_94_females_original_test_age_group_acc,
                    model_65_females_original_test_acc
                ])
print ("MODELS COMBINED")
'''
print ("STARTING TO COMBINE WITH WEIGHTS ON VALIDATION DATA")
combined_predictions_weighted_val = combineWeightedPredictions(val_i_vectors, val_data_padded,
                 [
                    dnn_6_females_predictions_val[1],
                    model_63_females_original_predictions_val,
                    model_94_females_original_predictions_val[1],
                    model_65_females_original_predictions_val
                ],
                [
                    dnn_6_females_original_val_age_acc,
                    model_63_females_original_val_acc,
                    model_94_females_original_val_age_group_acc,
                    model_65_females_original_val_acc
                ])
print ("STARTING TO COMBINE WITH WEIGHTS ON TEST DATA")
'''
combined_predictions_weighted_val = combineWeightedPredictions(test_i_vectors, test_data_padded,
                 [
                    dnn_6_females_predictions_test[1],
                    model_63_females_original_predictions_test,
                    model_94_females_original_predictions_test[1],
                    model_65_females_original_predictions_test
                ],
                [
                    dnn_6_females_original_test_age_acc,
                    model_63_females_original_test_acc,
                    model_94_females_original_test_age_group_acc,
                    model_65_females_original_test_acc
                ])
'''
print ("MODELS COMBINED")
'''
np.save('/home/hpc_lkpiel/predictions/val/combined_prediction_2.npy', combined_predictions_val)
print ("VAL SAVED")

np.save('/home/hpc_lkpiel/predictions/test/combined_prediction_2.npy', combined_predictions_test)
print ("TEST SAVED")
'''
np.save('/home/hpc_lkpiel/predictions/val/combined_prediction_weighted_2.npy', combined_predictions_weighted_val)
print ("VAL SAVED")

#np.save('/home/hpc_lkpiel/predictions/test/combined_prediction_weighted_2.npy', combined_predictions_weighted_val)
print ("TEST SAVED")

########################################