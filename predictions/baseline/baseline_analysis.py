#! /usr/bin/python3


import sys
print(sys.version)
import sys
import pandas
import numpy as np
from sklearn.metrics import mean_absolute_error
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

baseline_age_group_val = pandas.read_table("./val/age_group.txt", sep=" ")
baseline_age_group_test = pandas.read_table("./test/age_group.txt", sep=" ")


baseline_gender_val = pandas.read_table("./val/gender.txt", sep=" ")
baseline_gender_test = pandas.read_table("./test/gender.txt", sep=" ")



val_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/dev/export.csv", sep=" ")
val_data_i_vectors = format(val_data_i_vectors)
val_labels_age_group = onehot(val_data_i_vectors['AgeGroup'])
val_labels_gender = onehot(val_data_i_vectors['Gender'])

test_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/test/export.csv", sep=" ")
test_data_i_vectors = format(test_data_i_vectors)
test_labels_age_group = onehot(test_data_i_vectors['AgeGroup'])
test_labels_gender = onehot(test_data_i_vectors['Gender'])


def getAgeGroupModelAccuracy(predictions, data):
    correct_predictions = []
    for i in range(0, len(predictions)):
        if (predictions.iloc[i]['age_group'] == 'ag1' and np.argmax(data[i]) == 0):
            correct_predictions.append(1)
        elif (predictions.iloc[i]['age_group'] == 'ag2' and np.argmax(data[i]) == 1):
            correct_predictions.append(1)
        elif (predictions.iloc[i]['age_group'] == 'ag3' and np.argmax(data[i]) == 2):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    return correct_predictions.count(1)/len(predictions)


def getGenderModelAccuracy(predictions, real):
    correct_predictions = []

    for i in range(0, len(predictions)):
        if (predictions.iloc[i]['gender'] == 'm' and np.argmax(real[i]) == 0):
            correct_predictions.append(1)
        elif (predictions.iloc[i]['gender'] == 'f' and np.argmax(real[i]) == 1):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    return correct_predictions.count(1)/len(predictions)


print ("AGE GROUP")
print ("=======================")
print ("baseline accuracy on validation data: " + str(getAgeGroupModelAccuracy(baseline_age_group_val, val_labels_age_group)))
print ("baseline accuracy on test data: " + str(getAgeGroupModelAccuracy(baseline_age_group_test, test_labels_age_group)))
print ("=======================")

print ("GENDER")
print ("=======================")
print ("baseline accuracy on validation data: " + str(getGenderModelAccuracy(baseline_gender_val, val_labels_gender)))
print ("baseline accuracy on test data: " + str(getGenderModelAccuracy(baseline_gender_test, test_labels_gender)))
print ("=======================")


