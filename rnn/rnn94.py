#! /usr/bin/python3


import sys
print(sys.version)
import sys
import pandas
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Average, Merge,Input, Layer, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling2D, AveragePooling2D, Reshape, BatchNormalization
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
from keras.constraints import max_norm

import tensorflow as tf
import keras
from keras import backend as K



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


def age_group_accuracy(y_true, y_pred):
        array = np.array([0]*13 + [1]*2 + [2]*13000000)

        age_to_group = K.variable(value=array, dtype='int32', name='age_to_group')
        ages_true = tf.gather(age_to_group, tf.cast(tf.rint(y_true), tf.int32))
        ages_pred = tf.gather(age_to_group, tf.cast(tf.rint(y_pred), tf.int32))
        return K.mean(K.equal(ages_true, ages_pred), axis=-1)


#LOAD LABELS


train_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/train/export.csv", sep=" ")
train_data_i_vectors = format(train_data_i_vectors)
train_labels_age_group = onehot(train_data_i_vectors['AgeGroup'])
train_labels_age = train_data_i_vectors['Age']
train_labels_gender = onehot(train_data_i_vectors['Gender'])

#train_labels_age_group = np.load("../dnn/dnn_1_train_predictions.npy")

val_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/dev/export.csv", sep=" ")
val_data_i_vectors = format(val_data_i_vectors)
val_labels_age_group = onehot(val_data_i_vectors['AgeGroup'])
val_labels_age = val_data_i_vectors['Age']
val_labels_gender = onehot(val_data_i_vectors['Gender'])


test_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/test/export.csv", sep=" ")
test_data_i_vectors = format(test_data_i_vectors)
test_labels_age_group = onehot(test_data_i_vectors['AgeGroup'])

test_labels_age = test_data_i_vectors['Age']
test_labels_gender = onehot(test_data_i_vectors['Gender'])

print ("LABELS LOADED")


#LOAD DATA

#train_data_padded = np.load("/storage/hpc_lkpiel/data/fbank_train_data_padded.npy", encoding="bytes")
val_data_padded = np.load("/storage/hpc_lkpiel/data/fbank_val_data_padded.npy", encoding="bytes")
test_data_padded = np.load("/storage/hpc_lkpiel/data/fbank_test_data_padded.npy", encoding="bytes")
print ("DATA LOADED")

################################################################################################


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                              patience=2, min_lr=0.0001, verbose=1)


kernel_regularizer = regularizers.l2(0.0001)
input_layer = Input(shape=(1107, 20))
x =  Masking(mask_value=0.)(input_layer)
x =  Bidirectional(LSTM(64, return_sequences=True))(x)
x =  Bidirectional(LSTM(64, return_sequences=True))(x)
x = AttentionWithContext()(x)

output_layer_1 = Dense(2, activation='softmax', name='gender_output')(x)

y = Dense(100, activation='sigmoid')(x)
output_layer_2 =  Dense(1, name='age_output')(y)



model_94 = Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])

model_94.compile(loss={'gender_output':'categorical_crossentropy', 'age_output':'mse'},
                    optimizer=SGD(0.01),
                    metrics={'gender_output':'accuracy','age_output':age_group_accuracy})
print ("multi_model MODEL COMPILED")


checkpoint = ModelCheckpoint(filepath='/home/hpc_lkpiel/models/model_94.hdf5', monitor='val_age_output_age_group_accuracy', save_best_only=True)




'''
history = model_94.fit(x=train_data_padded,
        y=[train_labels_gender, train_labels_age],
        validation_data=(val_data_padded, [val_labels_gender, val_labels_age]),
        epochs=100,
        verbose=1,
        batch_size=256,
        callbacks=[checkpoint]
)

np.save('../history/history_model_94.npy', history.history)
modelHistory = np.load('../history/history_model_94.npy').item()

print ("HISTORY: ")
print (modelHistory)
print (model_94.evaluate(val_data_padded, [val_labels_gender, val_labels_age]))
print (model_94.evaluate(test_data_padded, [test_labels_gender, test_labels_age]))
'''
model_94.load_weights('/home/hpc_lkpiel/models/model_94.hdf5')

val_predictions = model_94.predict(val_data_padded)
print ("VAL PREDICTED")

test_predictions = model_94.predict(test_data_padded)
print ("TEST PREDICTED")

np.save('/home/hpc_lkpiel/predictions/val/model_94_gender.npy', val_predictions[0])
print ("VAL SAVED")

np.save('/home/hpc_lkpiel/predictions/test/model_94_gender.npy', test_predictions[0])
print ("TEST SAVED")

np.save('/home/hpc_lkpiel/predictions/val/model_94_age_group.npy', val_predictions[1])
print ("VAL SAVED")

np.save('/home/hpc_lkpiel/predictions/test/model_94_age_group.npy', test_predictions[1])
print ("TEST SAVED")

'''
valResult = model_94.evaluate(val_data_padded, val_labels_age_group)
testResult = model_94.evaluate(test_data_padded, test_labels_age_group)

file = open("results.txt","a")
file.write("\nmodel_94 VAL: " + str(valResult) + " TEST: " + str(testResult))
file.close()
'''
print ("WROTE TO FILE model_94")

########################################