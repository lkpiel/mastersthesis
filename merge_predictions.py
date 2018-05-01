#! /usr/bin/python3

import sys
import sys
import pandas
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

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


train_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/train/export.csv", sep=" ")
train_data_i_vectors = format(train_data_i_vectors)
train_labels_age_group = onehot(train_data_i_vectors['AgeGroup'])
train_labels_gender = onehot(train_data_i_vectors['Gender'])

#train_labels_age_group = np.load("../dnn/dnn_predicted_labels.npy")

val_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/dev/export.csv", sep=" ")
val_data_i_vectors = format(val_data_i_vectors)
val_labels_age_group = onehot(val_data_i_vectors['AgeGroup'])
val_labels_age_group_males = onehot(val_data_i_vectors['AgeGroup'][val_data_i_vectors['Gender'] == 0])
val_labels_age_group_females = onehot(val_data_i_vectors['AgeGroup'][val_data_i_vectors['Gender'] == 1])

val_labels_age = val_data_i_vectors['Age']
val_labels_gender = onehot(val_data_i_vectors['Gender'])


test_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/test/export.csv", sep=" ")
test_data_i_vectors = format(test_data_i_vectors)
test_labels_age_group = onehot(test_data_i_vectors['AgeGroup'])
test_labels_age_group_males = onehot(test_data_i_vectors['AgeGroup'][test_data_i_vectors['Gender'] == 0])
test_labels_age_group_females = onehot(test_data_i_vectors['AgeGroup'][test_data_i_vectors['Gender'] == 1])
print (len(test_data))
test_data_indexes_used_in_survey = []

for i in range(0, len(test_data) + 1):
    not_in_excluded = []
    for j in range(1, 10):
        if ('_' + str(j) + '_' not in test_data.iloc[i]['Utterance']):
            not_in_excluded.append(j)
    if (len(not_in_excluded) == 9):
        test_data_indexes_used_in_survey.append(i)


test_labels_gender = onehot(test_data_i_vectors['Gender'])
test_labels_age_group_used_in_survey = test_labels_age_group[test_data_indexes_used_in_survey]
test_labels_age_group_used_in_survey = test_labels_gender[test_data_indexes_used_in_survey]


print ("LABELS LOADED")

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


def getAgeGroupModelAccuracy(predictions, data):
    correct_predictions = []
    for i in range(0, len(predictions)):
        if (np.argmax(predictions[i]) == np.argmax(data[i])):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    return correct_predictions.count(1)/len(predictions)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference



# LOAD IN VAL PREDICTIONS
model_77_val = np.load("./val/model_77.npy", encoding="bytes")
dnn_1_val = np.load("./val/dnn_1.npy", encoding="bytes")
dnn_2_val = np.load("./val/dnn_2.npy", encoding="bytes")
dnn_4_val = np.load("./val/dnn_4_age_group.npy", encoding="bytes")
dnn_4_val_gender = np.load("./val/dnn_4_gender.npy", encoding="bytes")
dnn_4_males_original_val = np.load("./val/dnn_4_males_original.npy", encoding="bytes")
dnn_4_females_original_val = np.load("./val/dnn_4_females_original.npy", encoding="bytes")
model_87_val = np.load("./val/model_87_age_group.npy", encoding="bytes")
model_87_val_gender = np.load("./val/model_87_gender.npy", encoding="bytes")
model_87_males_original_val = np.load("./val/model_87_males_original.npy", encoding="bytes")
model_87_females_original_val = np.load("./val/model_87_females_original.npy", encoding="bytes")

model_86_val = np.load("./val/model_86.npy", encoding="bytes")
model_86_males_original_val = np.load("./val/model_86_males_original.npy", encoding="bytes")
model_86_females_original_val = np.load("./val/model_86_females_original.npy", encoding="bytes")
model_86_males_val = np.load("./val/model_86_males.npy", encoding="bytes")
model_86_females_val = np.load("./val/model_86_females.npy", encoding="bytes")

dnn_6_val_age_group = np.load("./val/dnn_6_age_group.npy", encoding="bytes")
dnn_6_val_age = np.load("./val/dnn_6_age.npy", encoding="bytes")
dnn_6_males_original_val_age_group = np.load("./val/dnn_6_age_group_males_original.npy", encoding="bytes")
dnn_6_males_original_val_age = np.load("./val/dnn_6_age_males_original.npy", encoding="bytes")
dnn_6_females_original_val_age_group = np.load("./val/dnn_6_age_group_females_original.npy", encoding="bytes")
dnn_6_females_original_val_age = np.load("./val/dnn_6_age_females_original.npy", encoding="bytes")

dnn_6_males_val_age_group = np.load("./val/dnn_6_age_group_males.npy", encoding="bytes")
dnn_6_males_val_age = np.load("./val/dnn_6_age_males.npy", encoding="bytes")
dnn_6_females_val_age_group = np.load("./val/dnn_6_age_group_females.npy", encoding="bytes")
dnn_6_females_val_age = np.load("./val/dnn_6_age_females.npy", encoding="bytes")
model_72_val = np.load("./val/model_72.npy", encoding="bytes")
model_72_males_original_val = np.load("./val/model_72_males_original.npy", encoding="bytes")
model_72_females_original_val = np.load("./val/model_72_females_original.npy", encoding="bytes")
model_72_males_val = np.load("./val/model_72_males.npy", encoding="bytes")
model_72_females_val = np.load("./val/model_72_females.npy", encoding="bytes")
model_57_val = np.load("./val/model_57.npy", encoding="bytes")
model_57_males_original_val = np.load("./val/model_57_males_original.npy", encoding="bytes")
model_57_females_original_val = np.load("./val/model_57_females_original.npy", encoding="bytes")
model_57_males_val = np.load("./val/model_57_males.npy", encoding="bytes")
model_57_females_val = np.load("./val/model_57_females.npy", encoding="bytes")
model_56_val = np.load("./val/model_56.npy", encoding="bytes")
model_56_males_original_val = np.load("./val/model_56_males_original.npy", encoding="bytes")
model_56_females_original_val = np.load("./val/model_56_females_original.npy", encoding="bytes")
model_67_val = np.load("./val/model_67.npy", encoding="bytes")
model_67_males_original_val = np.load("./val/model_67_males_original.npy", encoding="bytes")
model_67_females_original_val = np.load("./val/model_67_females_original.npy", encoding="bytes")
model_59_val = np.load("./val/model_59.npy", encoding="bytes")
model_69_val = np.load("./val/model_69.npy", encoding="bytes")
model_63_val = np.load("./val/model_63.npy", encoding="bytes")
model_63_males_original_val = np.load("./val/model_63_males_original.npy", encoding="bytes")
model_63_males_val = np.load("./val/model_63_males.npy", encoding="bytes")
model_63_females_original_val = np.load("./val/model_63_females_original.npy", encoding="bytes")
model_63_females_val = np.load("./val/model_63_females.npy", encoding="bytes")
dnn_7_males_original_age_group_val = np.load("./val/dnn_7_males_original_age_group.npy", encoding="bytes")
dnn_7_females_original_age_group_val = np.load("./val/dnn_7_females_original_age_group.npy", encoding="bytes")
dnn_7_val_age_group= np.load("./val/dnn_7_age_group.npy", encoding="bytes")
dnn_7_val_gender= np.load("./val/dnn_7_gender.npy", encoding="bytes")
model_23_val = np.load("./val/model_23.npy", encoding="bytes")
model_73_val = np.load("./val/model_73.npy", encoding="bytes")
model_38_val = np.load("./val/model_38.npy", encoding="bytes")
model_61_val = np.load("./val/model_61.npy", encoding="bytes")
dnn_3_val = np.load("./val/dnn_3_gender.npy", encoding="bytes")
gender_1_val = np.load("./val/gender_1.npy", encoding="bytes")
gender_8_val = np.load("./val/gender_8.npy", encoding="bytes")
gender_4_val = np.load("./val/gender_4.npy", encoding="bytes")
gender_5_val = np.load("./val/gender_5.npy", encoding="bytes")
gender_6_val = np.load("./val/gender_6.npy", encoding="bytes")
gender_2_val = np.load("./val/gender_2.npy", encoding="bytes")
gender_9_val = np.load("./val/gender_9.npy", encoding="bytes")
model_90_val = np.load("./val/model_90.npy", encoding="bytes")

model_71_val = np.load("./val/model_71.npy", encoding="bytes")
model_62_val = np.load("./val/model_62.npy", encoding="bytes")

model_68_val = np.load("./val/model_68.npy", encoding="bytes")
model_68_males_original_val = np.load("./val/model_68_males_original.npy", encoding="bytes")
model_68_females_original_val = np.load("./val/model_68_females_original.npy", encoding="bytes")
model_68_males_val = np.load("./val/model_68_males.npy", encoding="bytes")
model_68_females_val = np.load("./val/model_68_females.npy", encoding="bytes")

model_58_val = np.load("./val/model_58.npy", encoding="bytes")
model_65_val = np.load("./val/model_65.npy", encoding="bytes")
model_65_males_original_val = np.load("./val/model_65_males_original.npy", encoding="bytes")
model_65_males_val = np.load("./val/model_65_males.npy", encoding="bytes")
model_65_females_original_val = np.load("./val/model_65_females_original.npy", encoding="bytes")
model_65_females_val = np.load("./val/model_65_females.npy", encoding="bytes")

model_41_val = np.load("./val/model_41.npy", encoding="bytes")
model_39_val = np.load("./val/model_39.npy", encoding="bytes")
model_75_val = np.load("./val/model_75.npy", encoding="bytes")
model_75_males_original_val = np.load("./val/model_75_males_original.npy", encoding="bytes")
model_75_females_original_val = np.load("./val/model_75_females_original.npy", encoding="bytes")
model_64_val = np.load("./val/model_64.npy", encoding="bytes")
model_58_val = np.load("./val/model_58.npy", encoding="bytes")
model_42_val = np.load("./val/model_42.npy", encoding="bytes")
model_42_males_original_val = np.load("./val/model_42_males_original.npy", encoding="bytes")
model_42_females_original_val = np.load("./val/model_42_females_original.npy", encoding="bytes")
model_66_val = np.load("./val/model_66.npy", encoding="bytes")
model_66_males_original_val = np.load("./val/model_66_males_original.npy", encoding="bytes")
model_66_females_original_val = np.load("./val/model_66_females_original.npy", encoding="bytes")
model_37_val = np.load("./val/model_37.npy", encoding="bytes")

dnn_2_males_original_val = np.load("./val/dnn_2_males_original.npy", encoding="bytes")
dnn_2_males_val = np.load("./val/dnn_2_males.npy", encoding="bytes")
dnn_2_females_original_val = np.load("./val/dnn_2_females_original.npy", encoding="bytes")
dnn_2_females_val = np.load("./val/dnn_2_females.npy", encoding="bytes")
dnn_1_males_loss_val = np.load("./val/dnn_1_males_loss.npy", encoding="bytes")
dnn_1_males_original_val = np.load("./val/dnn_1_males_original.npy", encoding="bytes")
dnn_1_males_val = np.load("./val/dnn_1_males.npy", encoding="bytes")
dnn_1_females_original_val = np.load("./val/dnn_1_females_original.npy", encoding="bytes")
dnn_1_females_val = np.load("./val/dnn_1_females.npy", encoding="bytes")

model_93_val_age_group = np.load("./val/model_93_age_group.npy", encoding="bytes")
model_93_val_gender = np.load("./val/model_93_gender.npy", encoding="bytes")

model_95_val_age_group = np.load("./val/model_95_age_group.npy", encoding="bytes")
model_95_val_gender = np.load("./val/model_95_gender.npy", encoding="bytes")

#model_91_val_gender = np.load("./val/model_91_gender.npy", encoding="bytes")
model_88_val_gender = np.load("./val/model_88_gender.npy", encoding="bytes")
model_88_val_age_group = np.load("./val/model_88_age_group.npy", encoding="bytes")
model_88_males_original_val_age_group = np.load("./val/model_88_males_original_age_group.npy", encoding="bytes")
model_88_females_original_val_age_group = np.load("./val/model_88_females_original_age_group.npy", encoding="bytes")

model_53_val_age = np.load("./val/model_53_age.npy", encoding="bytes")
model_53_val_age_group = np.load("./val/model_53_age_group.npy", encoding="bytes")
model_31_val = np.load("./val/model_31.npy", encoding="bytes")
model_48_val = np.load("./val/model_48.npy", encoding="bytes")
model_43_val = np.load("./val/model_43.npy", encoding="bytes")
model_43_males_original_val = np.load("./val/model_43_males_original.npy", encoding="bytes")
model_43_females_original_val = np.load("./val/model_43_females_original.npy", encoding="bytes")
model_43_males_val = np.load("./val/model_43_males.npy", encoding="bytes")
model_43_females_val = np.load("./val/model_43_females.npy", encoding="bytes")
model_84_val = np.load("./val/model_84.npy", encoding="bytes")
model_85_val = np.load("./val/model_85.npy", encoding="bytes")
model_26_val = np.load("./val/model_26.npy", encoding="bytes")
model_82_val = np.load("./val/model_82.npy", encoding="bytes")
model_51_val = np.load("./val/model_51.npy", encoding="bytes")
model_54_val = np.load("./val/model_54.npy", encoding="bytes")

model_91_val_gender = np.load("./val/model_91_gender.npy", encoding="bytes")
model_91_val_age_group = np.load("./val/model_91_age_group.npy", encoding="bytes")
model_94_val_gender = np.load("./val/model_94_gender.npy", encoding="bytes")
model_94_val_age_group = np.load("./val/model_94_age_group.npy", encoding="bytes")
model_94_males_original_val_age_group = np.load("./val/model_94_males_original_age_group.npy", encoding="bytes")
model_94_females_original_val_age_group = np.load("./val/model_94_females_original_age_group.npy", encoding="bytes")

# LOAD IN TEST PREDICTIONS
model_77_test = np.load("./test/model_77.npy", encoding="bytes")
dnn_1_test = np.load("./test/dnn_1.npy", encoding="bytes")
dnn_2_test = np.load("./test/dnn_2.npy", encoding="bytes")
dnn_4_test = np.load("./test/dnn_4_age_group.npy", encoding="bytes")
dnn_4_test_gender = np.load("./test/dnn_4_gender.npy", encoding="bytes")
dnn_4_males_original_test = np.load("./test/dnn_4_males_original.npy", encoding="bytes")
dnn_4_females_original_test = np.load("./test/dnn_4_females_original.npy", encoding="bytes")
model_87_test = np.load("./test/model_87_age_group.npy", encoding="bytes")
model_87_test_gender = np.load("./test/model_87_gender.npy", encoding="bytes")
model_87_males_original_test = np.load("./test/model_87_males_original.npy", encoding="bytes")
model_87_females_original_test = np.load("./test/model_87_females_original.npy", encoding="bytes")

model_86_test = np.load("./test/model_86.npy", encoding="bytes")
model_86_males_original_test = np.load("./test/model_86_males_original.npy", encoding="bytes")
model_86_females_original_test = np.load("./test/model_86_females_original.npy", encoding="bytes")
model_86_males_test = np.load("./test/model_86_males.npy", encoding="bytes")
model_86_females_test = np.load("./test/model_86_females.npy", encoding="bytes")
dnn_6_test_age_group = np.load("./test/dnn_6_age_group.npy", encoding="bytes")
dnn_6_test_age = np.load("./test/dnn_6_age.npy", encoding="bytes")
dnn_6_males_original_test_age_group = np.load("./test/dnn_6_age_group_males_original.npy", encoding="bytes")
dnn_6_males_original_test_age = np.load("./test/dnn_6_age_males_original.npy", encoding="bytes")
dnn_6_females_original_test_age_group = np.load("./test/dnn_6_age_group_females_original.npy", encoding="bytes")
dnn_6_females_original_test_age = np.load("./test/dnn_6_age_females_original.npy", encoding="bytes")

dnn_6_males_test_age_group = np.load("./test/dnn_6_age_group_males.npy", encoding="bytes")
dnn_6_males_test_age = np.load("./test/dnn_6_age_males.npy", encoding="bytes")
dnn_6_females_test_age_group = np.load("./test/dnn_6_age_group_females.npy", encoding="bytes")
dnn_6_females_test_age = np.load("./test/dnn_6_age_females.npy", encoding="bytes")
model_72_test = np.load("./test/model_72.npy", encoding="bytes")
model_72_males_original_test = np.load("./test/model_72_males_original.npy", encoding="bytes")
model_72_females_original_test = np.load("./test/model_72_females_original.npy", encoding="bytes")
model_72_males_test = np.load("./test/model_72_males.npy", encoding="bytes")
model_72_females_test = np.load("./test/model_72_females.npy", encoding="bytes")
model_57_test = np.load("./test/model_57.npy", encoding="bytes")
model_57_males_original_test = np.load("./test/model_57_males_original.npy", encoding="bytes")
model_57_females_original_test = np.load("./test/model_57_females_original.npy", encoding="bytes")
model_57_males_test = np.load("./test/model_57_males.npy", encoding="bytes")
model_57_females_test = np.load("./test/model_57_females.npy", encoding="bytes")
model_56_test = np.load("./test/model_56.npy", encoding="bytes")

model_56_males_original_test = np.load("./test/model_56_males_original.npy", encoding="bytes")
model_56_females_original_test = np.load("./test/model_56_females_original.npy", encoding="bytes")
model_67_test = np.load("./test/model_67.npy", encoding="bytes")
model_67_males_original_test = np.load("./test/model_67_males_original.npy", encoding="bytes")
model_67_females_original_test = np.load("./test/model_67_females_original.npy", encoding="bytes")
model_59_test = np.load("./test/model_59.npy", encoding="bytes")
model_69_test = np.load("./test/model_69.npy", encoding="bytes")
model_63_test = np.load("./test/model_63.npy", encoding="bytes")
model_63_males_original_test = np.load("./test/model_63_males_original.npy", encoding="bytes")
model_63_males_test = np.load("./test/model_63_males.npy", encoding="bytes")
model_63_females_original_test = np.load("./test/model_63_females_original.npy", encoding="bytes")
model_63_females_test = np.load("./test/model_63_females.npy", encoding="bytes")
dnn_7_males_original_age_group_test = np.load("./test/dnn_7_males_original_age_group.npy", encoding="bytes")
dnn_7_females_original_age_group_test = np.load("./test/dnn_7_females_original_age_group.npy", encoding="bytes")
dnn_7_test_age_group = np.load("./test/dnn_7_age_group.npy", encoding="bytes")
dnn_7_test_gender = np.load("./test/dnn_7_gender.npy", encoding="bytes")
model_23_test = np.load("./test/model_23.npy", encoding="bytes")
model_73_test = np.load("./test/model_73.npy", encoding="bytes")
model_38_test = np.load("./test/model_38.npy", encoding="bytes")
model_61_test = np.load("./test/model_61.npy", encoding="bytes")
dnn_3_test = np.load("./test/dnn_3_gender.npy", encoding="bytes")
gender_1_test = np.load("./test/gender_1.npy", encoding="bytes")
gender_8_test = np.load("./test/gender_8.npy", encoding="bytes")
gender_4_test = np.load("./test/gender_4.npy", encoding="bytes")
gender_5_test = np.load("./test/gender_5.npy", encoding="bytes")
gender_6_test = np.load("./test/gender_6.npy", encoding="bytes")
gender_2_test = np.load("./test/gender_2.npy", encoding="bytes")
gender_9_test = np.load("./test/gender_9.npy", encoding="bytes")
model_90_test = np.load("./test/model_90.npy", encoding="bytes")
model_71_test = np.load("./test/model_71.npy", encoding="bytes")
model_62_test = np.load("./test/model_62.npy", encoding="bytes")
model_68_test = np.load("./test/model_68.npy", encoding="bytes")
model_68_males_original_test = np.load("./test/model_68_males_original.npy", encoding="bytes")
model_68_females_original_test = np.load("./test/model_68_females_original.npy", encoding="bytes")
model_68_males_test = np.load("./test/model_68_males.npy", encoding="bytes")
model_68_females_test = np.load("./test/model_68_females.npy", encoding="bytes")
model_58_test = np.load("./test/model_58.npy", encoding="bytes")
model_65_test = np.load("./test/model_65.npy", encoding="bytes")
model_65_males_original_test = np.load("./test/model_65_males_original.npy", encoding="bytes")
model_65_males_test = np.load("./test/model_65_males.npy", encoding="bytes")
model_65_females_original_test = np.load("./test/model_65_females_original.npy", encoding="bytes")
model_65_females_test = np.load("./test/model_65_females.npy", encoding="bytes")

model_58_test = np.load("./test/model_58.npy", encoding="bytes")
model_42_test = np.load("./test/model_42.npy", encoding="bytes")
model_42_males_original_test = np.load("./test/model_42_males_original.npy", encoding="bytes")
model_42_females_original_test = np.load("./test/model_42_females_original.npy", encoding="bytes")
model_66_test = np.load("./test/model_66.npy", encoding="bytes")
model_66_males_original_test = np.load("./test/model_66_males_original.npy", encoding="bytes")
model_66_females_original_test = np.load("./test/model_66_females_original.npy", encoding="bytes")
model_37_test = np.load("./test/model_37.npy", encoding="bytes")

model_41_test = np.load("./test/model_41.npy", encoding="bytes")
model_39_test = np.load("./test/model_39.npy", encoding="bytes")
model_75_test = np.load("./test/model_75.npy", encoding="bytes")
model_75_males_original_test = np.load("./test/model_75_males_original.npy", encoding="bytes")
model_75_females_original_test = np.load("./test/model_75_females_original.npy", encoding="bytes")
model_64_test = np.load("./test/model_64.npy", encoding="bytes")
dnn_2_males_original_test = np.load("./test/dnn_2_males_original.npy", encoding="bytes")
dnn_2_males_test = np.load("./test/dnn_2_males.npy", encoding="bytes")
dnn_2_females_original_test = np.load("./test/dnn_2_females_original.npy", encoding="bytes")
dnn_2_females_test = np.load("./test/dnn_2_females.npy", encoding="bytes")
dnn_1_males_loss_test = np.load("./test/dnn_1_males_loss.npy", encoding="bytes")
dnn_1_males_original_test = np.load("./test/dnn_1_males_original.npy", encoding="bytes")
dnn_1_males_test = np.load("./test/dnn_1_males.npy", encoding="bytes")
dnn_1_females_original_test = np.load("./test/dnn_1_females_original.npy", encoding="bytes")
dnn_1_females_test = np.load("./test/dnn_1_females.npy", encoding="bytes")


model_93_test_age_group = np.load("./test/model_93_age_group.npy", encoding="bytes")
model_93_test_gender = np.load("./test/model_93_gender.npy", encoding="bytes")
model_95_test_age_group = np.load("./test/model_95_age_group.npy", encoding="bytes")
model_95_test_gender = np.load("./test/model_95_gender.npy", encoding="bytes")

#model_91_test_gender = np.load("./test/model_91_gender.npy", encoding="bytes")
model_88_test_gender = np.load("./test/model_88_gender.npy", encoding="bytes")
model_88_test_age_group = np.load("./test/model_88_age_group.npy", encoding="bytes")
model_88_males_original_test_age_group = np.load("./test/model_88_males_original_age_group.npy", encoding="bytes")
model_88_females_original_test_age_group = np.load("./test/model_88_females_original_age_group.npy", encoding="bytes")

model_53_test_age = np.load("./test/model_53_age.npy", encoding="bytes")
model_53_test_age_group = np.load("./test/model_53_age_group.npy", encoding="bytes")
model_31_test = np.load("./test/model_31.npy", encoding="bytes")
model_48_test = np.load("./test/model_48.npy", encoding="bytes")

model_43_test = np.load("./test/model_43.npy", encoding="bytes")
model_43_males_original_test = np.load("./test/model_43_males_original.npy", encoding="bytes")
model_43_females_original_test = np.load("./test/model_43_females_original.npy", encoding="bytes")
model_43_males_test = np.load("./test/model_43_males.npy", encoding="bytes")
model_43_females_test = np.load("./test/model_43_females.npy", encoding="bytes")

model_84_test = np.load("./test/model_84.npy", encoding="bytes")
model_85_test = np.load("./test/model_85.npy", encoding="bytes")
model_26_test = np.load("./test/model_26.npy", encoding="bytes")
model_82_test = np.load("./test/model_82.npy", encoding="bytes")
model_51_test = np.load("./test/model_51.npy", encoding="bytes")
model_54_test = np.load("./test/model_54.npy", encoding="bytes")

model_91_test_gender = np.load("./test/model_91_gender.npy", encoding="bytes")
model_91_test_age_group = np.load("./test/model_91_age_group.npy", encoding="bytes")
model_94_test_gender = np.load("./test/model_94_gender.npy", encoding="bytes")
model_94_test_age_group = np.load("./test/model_94_age_group.npy", encoding="bytes")
model_94_males_original_test_age_group = np.load("./test/model_94_males_original_age_group.npy", encoding="bytes")
model_94_females_original_test_age_group = np.load("./test/model_94_females_original_age_group.npy", encoding="bytes")

combined_predictions_val = np.load("./val/combined_prediction.npy", encoding="bytes")
combined_predictions_weighted_val = np.load("./val/combined_prediction_weighted.npy", encoding="bytes")
combined_predictions_test = np.load("./test/combined_prediction.npy", encoding="bytes")
combined_predictions_weighted_test = np.load("./test/combined_prediction_weighted.npy", encoding="bytes")


combined_predictions_val_2 = np.load("./val/combined_prediction_2.npy", encoding="bytes")
combined_predictions_weighted_val_2 = np.load("./val/combined_prediction_weighted_2.npy", encoding="bytes")
combined_predictions_test_2 = np.load("./test/combined_prediction_2.npy", encoding="bytes")
combined_predictions_weighted_test_2 = np.load("./test/combined_prediction_weighted_2.npy", encoding="bytes")



#merged_predictions = (dnn_2_val + model_77_val)/2
dnn_2_val = np.asarray(formatAgePredictions(dnn_2_val))
dnn_2_test = np.asarray(formatAgePredictions(dnn_2_test))
dnn_6_val_age = np.asarray(formatAgePredictions(dnn_6_val_age))
dnn_6_test_age = np.asarray(formatAgePredictions(dnn_6_test_age))
dnn_6_males_original_val_age =  np.asarray(formatAgePredictions(dnn_6_males_original_val_age))
dnn_6_females_original_val_age =  np.asarray(formatAgePredictions(dnn_6_females_original_val_age))
dnn_6_males_original_test_age =  np.asarray(formatAgePredictions(dnn_6_males_original_test_age))
dnn_6_females_original_test_age =  np.asarray(formatAgePredictions(dnn_6_females_original_test_age))

dnn_6_males_val_age =  np.asarray(formatAgePredictions(dnn_6_males_val_age))
dnn_6_females_val_age =  np.asarray(formatAgePredictions(dnn_6_females_val_age))
dnn_6_males_test_age =  np.asarray(formatAgePredictions(dnn_6_males_test_age))
dnn_6_females_test_age =  np.asarray(formatAgePredictions(dnn_6_females_test_age))
dnn_7_val_age_group = np.asarray(formatAgePredictions(dnn_7_val_age_group))
dnn_7_males_original_age_group_val = np.asarray(formatAgePredictions(dnn_7_males_original_age_group_val))
dnn_7_females_original_age_group_val = np.asarray(formatAgePredictions(dnn_7_females_original_age_group_val))
dnn_7_test_age_group = np.asarray(formatAgePredictions(dnn_7_test_age_group))
dnn_7_males_original_age_group_test = np.asarray(formatAgePredictions(dnn_7_males_original_age_group_test))
dnn_7_females_original_age_group_test = np.asarray(formatAgePredictions(dnn_7_females_original_age_group_test))
dnn_2_males_val = np.asarray(formatAgePredictions(dnn_2_males_val))
dnn_2_males_original_val = np.asarray(formatAgePredictions(dnn_2_males_original_val))
dnn_2_males_test = np.asarray(formatAgePredictions(dnn_2_males_test))
dnn_2_males_original_test = np.asarray(formatAgePredictions(dnn_2_males_original_test))
dnn_2_females_val = np.asarray(formatAgePredictions(dnn_2_females_val))
dnn_2_females_original_val = np.asarray(formatAgePredictions(dnn_2_females_original_val))
dnn_2_females_test = np.asarray(formatAgePredictions(dnn_2_females_test))
dnn_2_females_original_test = np.asarray(formatAgePredictions(dnn_2_females_original_test))
model_93_val_age_group = np.asarray(formatAgePredictions(model_93_val_age_group))
model_93_test_age_group = np.asarray(formatAgePredictions(model_93_test_age_group))
model_95_val_age_group = np.asarray(formatAgePredictions(model_95_val_age_group))
model_95_test_age_group = np.asarray(formatAgePredictions(model_95_test_age_group))
model_53_val_age = np.asarray(formatAgePredictions(model_53_val_age))
model_53_test_age  = np.asarray(formatAgePredictions(model_53_test_age))
model_48_val = np.asarray(formatAgePredictions(model_48_val))
model_48_test = np.asarray(formatAgePredictions(model_48_test))
model_82_val= np.asarray(formatAgePredictions(model_82_val))
model_82_test= np.asarray(formatAgePredictions(model_82_test))
model_51_val= np.asarray(formatAgePredictions(model_51_val))
model_51_test= np.asarray(formatAgePredictions(model_51_test))
model_54_val= np.asarray(formatAgePredictions(model_54_val))
model_54_test= np.asarray(formatAgePredictions(model_54_test))
model_91_val_age_group= np.asarray(formatAgePredictions(model_91_val_age_group))
model_91_test_age_group= np.asarray(formatAgePredictions(model_91_test_age_group))
model_94_val_age_group = np.asarray(formatAgePredictions(model_94_val_age_group))
model_94_test_age_group= np.asarray(formatAgePredictions(model_94_test_age_group))
model_94_males_original_val_age_group = np.asarray(formatAgePredictions(model_94_males_original_val_age_group))
model_94_females_original_val_age_group= np.asarray(formatAgePredictions(model_94_females_original_val_age_group))
model_94_males_original_test_age_group= np.asarray(formatAgePredictions(model_94_males_original_test_age_group))
model_94_females_original_test_age_group= np.asarray(formatAgePredictions(model_94_females_original_test_age_group))


'''
print ("\nVALIDATION ACCURACIES")
print ("dnn_1_val accuracy " + str(getAgeGroupModelAccuracy(dnn_1_val, val_labels_age_group)))
print ("dnn_2_val accuracy " + str(getAgeGroupModelAccuracy(dnn_2_val, val_labels_age_group)))
print ("dnn_4_val_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_4_val, val_labels_age_group)))
print ("dnn_4_val_gender accuracy " + str(getAgeGroupModelAccuracy(dnn_4_val_gender, val_labels_gender)))
print ("dnn_4_males_original_val accuracy " + str(getAgeGroupModelAccuracy(dnn_4_males_original_val, val_labels_age_group_males)))
print ("dnn_4_females_original_val accuracy " + str(getAgeGroupModelAccuracy(dnn_4_females_original_val, val_labels_age_group_females)))

print ("model_77_val accuracy " + str(getAgeGroupModelAccuracy(model_77_val, val_labels_age_group)))
print ("model_87_val accuracy " + str(getAgeGroupModelAccuracy(model_87_val, val_labels_age_group)))
print ("model_87_val_gender accuracy " + str(getAgeGroupModelAccuracy(model_87_val_gender, val_labels_gender)))
print ("model_87_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_87_males_original_val, val_labels_age_group_males)))
print ("model_87_females__original_val accuracy " + str(getAgeGroupModelAccuracy(model_87_females_original_val, val_labels_age_group_females)))

print ("model_86_val accuracy " + str(getAgeGroupModelAccuracy(model_86_val, val_labels_age_group)))
print ("model_86_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_86_males_original_val, val_labels_age_group_males)))
print ("model_86_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_86_females_original_val, val_labels_age_group_females)))
print ("model_86_males_val accuracy " + str(getAgeGroupModelAccuracy(model_86_males_val, val_labels_age_group_males)))
print ("model_86_females_val accuracy " + str(getAgeGroupModelAccuracy(model_86_females_val, val_labels_age_group_females)))

print ("dnn_6_val_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_6_val_age_group, val_labels_age_group)))
print ("dnn_6_val_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_val_age, val_labels_age_group)))
print ("model_72_val accuracy " + str(getAgeGroupModelAccuracy(model_72_val, val_labels_age_group)))
print ("model_72_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_72_males_original_val, val_labels_age_group_males)))
print ("model_72_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_72_females_original_val, val_labels_age_group_females)))
print ("model_72_males_val accuracy " + str(getAgeGroupModelAccuracy(model_72_males_val, val_labels_age_group_males)))
print ("model_72_females_val accuracy " + str(getAgeGroupModelAccuracy(model_72_females_val, val_labels_age_group_females)))
print ("model_57_val accuracy " + str(getAgeGroupModelAccuracy(model_57_val, val_labels_age_group)))
print ("model_57_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_57_males_original_val, val_labels_age_group_males)))
print ("model_57_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_57_females_original_val, val_labels_age_group_females)))
print ("model_57_males_val accuracy " + str(getAgeGroupModelAccuracy(model_57_males_val, val_labels_age_group_males)))
print ("model_57_females_val accuracy " + str(getAgeGroupModelAccuracy(model_57_females_val, val_labels_age_group_females)))
print ("model_56_val accuracy " + str(getAgeGroupModelAccuracy(model_56_val, val_labels_age_group)))
print ("model_56_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_56_males_original_val, val_labels_age_group_males)))
print ("model_56_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_56_females_original_val, val_labels_age_group_females)))
print ("model_67_val accuracy " + str(getAgeGroupModelAccuracy(model_67_val, val_labels_age_group)))
print ("model_59_val accuracy " + str(getAgeGroupModelAccuracy(model_59_val, val_labels_age_group)))
print ("model_69_val accuracy " + str(getAgeGroupModelAccuracy(model_69_val, val_labels_age_group)))
print ("model_63_val accuracy " + str(getAgeGroupModelAccuracy(model_63_val, val_labels_age_group)))
print ("model_63_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_63_males_original_val, val_labels_age_group_males)))
print ("model_63_males_val accuracy " + str(getAgeGroupModelAccuracy(model_63_males_val, val_labels_age_group_males)))
print ("model_63_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_63_females_original_val, val_labels_age_group_females)))
print ("model_63_females_val accuracy " + str(getAgeGroupModelAccuracy(model_63_females_val, val_labels_age_group_females)))
print ("dnn_7_val_gender accuracy " + str(getAgeGroupModelAccuracy(dnn_7_val_gender, val_labels_gender)))
print ("dnn_7_val_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_7_val_age_group, val_labels_age_group)))
print ("dnn_7_males_original_age_group_val accuracy " + str(getAgeGroupModelAccuracy(dnn_7_males_original_age_group_val, val_labels_age_group_males)))
print ("dnn_7_females_original_age_group_val accuracy " + str(getAgeGroupModelAccuracy(dnn_7_females_original_age_group_val, val_labels_age_group_females)))
print ("model_23_val accuracy " + str(getAgeGroupModelAccuracy(model_23_val, val_labels_age_group)))
print ("model_73_val accuracy " + str(getAgeGroupModelAccuracy(model_73_val, val_labels_age_group)))
print ("model_38_val accuracy " + str(getAgeGroupModelAccuracy(model_38_val, val_labels_age_group)))
print ("model_61_val accuracy " + str(getAgeGroupModelAccuracy(model_61_val, val_labels_age_group)))
print ("   dnn_3_val accuracy " + str(getAgeGroupModelAccuracy(dnn_3_val, val_labels_gender)))
print ("gender_1_val accuracy " + str(getAgeGroupModelAccuracy(gender_1_val, val_labels_gender)))
print ("gender_8_val accuracy " + str(getAgeGroupModelAccuracy(gender_8_val, val_labels_gender)))
print ("gender_4_val accuracy " + str(getAgeGroupModelAccuracy(gender_4_val, val_labels_gender)))
print ("gender_5_val accuracy " + str(getAgeGroupModelAccuracy(gender_5_val, val_labels_gender)))
print ("gender_6_val accuracy " + str(getAgeGroupModelAccuracy(gender_6_val, val_labels_gender)))
print ("gender_2_val accuracy " + str(getAgeGroupModelAccuracy(gender_2_val, val_labels_gender)))
print ("gender_9_val accuracy " + str(getAgeGroupModelAccuracy(gender_9_val, val_labels_gender)))
print ("model_90_val accuracy " + str(getAgeGroupModelAccuracy(model_90_val, val_labels_gender)))
print ("model_71_val accuracy " + str(getAgeGroupModelAccuracy(model_71_val, val_labels_age_group)))
print ("model_62_val accuracy " + str(getAgeGroupModelAccuracy(model_62_val, val_labels_age_group)))
print ("model_68_val accuracy " + str(getAgeGroupModelAccuracy(model_68_val, val_labels_age_group)))
print ("model_68_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_68_males_original_val, val_labels_age_group_males)))
print ("model_68_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_68_females_original_val, val_labels_age_group_females)))
print ("model_68_males_val accuracy " + str(getAgeGroupModelAccuracy(model_68_males_val, val_labels_age_group_males)))
print ("model_68_females_val accuracy " + str(getAgeGroupModelAccuracy(model_68_females_val, val_labels_age_group_females)))

print ("model_58_val accuracy " + str(getAgeGroupModelAccuracy(model_58_val, val_labels_age_group)))
print ("model_65_val accuracy " + str(getAgeGroupModelAccuracy(model_65_val, val_labels_age_group)))
print ("model_65_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_65_males_original_val, val_labels_age_group_males)))
print ("model_65_males_val accuracy " + str(getAgeGroupModelAccuracy(model_65_males_val, val_labels_age_group_males)))

print ("model_65_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_65_females_original_val, val_labels_age_group_females)))
print ("model_65_females_val accuracy " + str(getAgeGroupModelAccuracy(model_65_females_val, val_labels_age_group_females)))

print ("model_58_val accuracy " + str(getAgeGroupModelAccuracy(model_58_val, val_labels_age_group)))
print ("model_41_val accuracy " + str(getAgeGroupModelAccuracy(model_41_val, val_labels_age_group)))
print ("model_39_val accuracy " + str(getAgeGroupModelAccuracy(model_39_val, val_labels_age_group)))
print ("model_75_val accuracy " + str(getAgeGroupModelAccuracy(model_75_val, val_labels_age_group)))
print ("model_75_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_75_males_original_val, val_labels_age_group_males)))
print ("model_75_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_75_females_original_val, val_labels_age_group_females)))
print ("model_64_val accuracy " + str(getAgeGroupModelAccuracy(model_64_val, val_labels_age_group)))
print ("dnn_2_males_original_val accuracy " + str(getAgeGroupModelAccuracy(dnn_2_males_original_val, val_labels_age_group_males)))
print ("dnn_2_males_val accuracy " + str(getAgeGroupModelAccuracy(dnn_2_males_val, val_labels_age_group_males)))
print ("dnn_2_females_original_val accuracy " + str(getAgeGroupModelAccuracy(dnn_2_females_original_val, val_labels_age_group_females)))
print ("dnn_2_females_val accuracy " + str(getAgeGroupModelAccuracy(dnn_2_females_val, val_labels_age_group_females)))
print ("dnn_1_males_loss_val accuracy " + str(getAgeGroupModelAccuracy(dnn_1_males_loss_val, val_labels_age_group_males)))
print ("dnn_1_males_original_val accuracy " + str(getAgeGroupModelAccuracy(dnn_1_males_original_val, val_labels_age_group_males)))
print ("dnn_1_males_val accuracy " + str(getAgeGroupModelAccuracy(dnn_1_males_val, val_labels_age_group_males)))
print ("dnn_1_females_original_val accuracy " + str(getAgeGroupModelAccuracy(dnn_1_females_original_val, val_labels_age_group_females)))
print ("dnn_1_females_val accuracy " + str(getAgeGroupModelAccuracy(dnn_1_females_val, val_labels_age_group_females)))
print ("model_42_val accuracy " + str(getAgeGroupModelAccuracy(model_42_val, val_labels_age_group)))
print ("model_42_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_42_males_original_val, val_labels_age_group_males)))
print ("model_42_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_42_females_original_val, val_labels_age_group_females)))
print ("model_66_val accuracy " + str(getAgeGroupModelAccuracy(model_66_val, val_labels_age_group)))
print ("model_66_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_66_males_original_val, val_labels_age_group_males)))
print ("model_66_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_66_females_original_val, val_labels_age_group_females)))
print ("model_37_val accuracy " + str(getAgeGroupModelAccuracy(model_37_val, val_labels_age_group)))
print ("model_82_val accuracy " + str(getAgeGroupModelAccuracy(model_82_val, val_labels_age_group)))

print ("dnn_6_males_original_val_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_6_males_original_val_age_group, val_labels_age_group_males)))
print ("dnn_6_females_original_val_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_6_females_original_val_age_group, val_labels_age_group_females)))
print ("dnn_6_males_original_val_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_males_original_val_age, val_labels_age_group_males)))
print ("dnn_6_females_original_val_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_females_original_val_age, val_labels_age_group_females)))

print ("dnn_6_males_val_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_6_males_val_age_group, val_labels_age_group_males)))
print ("dnn_6_females_val_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_6_females_val_age_group, val_labels_age_group_females)))
print ("dnn_6_males_val_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_males_val_age, val_labels_age_group_males)))
print ("dnn_6_females_val_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_females_val_age, val_labels_age_group_females)))


print ("model_67_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_67_males_original_val, val_labels_age_group_males)))
print ("model_67_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_67_females_original_val, val_labels_age_group_females)))

print ("model_93_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_93_val_age_group, val_labels_age_group)))
print ("model_93_val_gender accuracy " + str(getAgeGroupModelAccuracy(model_93_val_gender, val_labels_gender)))

print ("model_95_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_95_val_age_group, val_labels_age_group)))
print ("model_95_val_gender accuracy " + str(getAgeGroupModelAccuracy(model_95_val_gender, val_labels_gender)))


#print ("model_91_val_gender accuracy " + str(getAgeGroupModelAccuracy(model_91_val_gender, val_labels_gender)))
print ("model_88_val_gender accuracy " + str(getAgeGroupModelAccuracy(model_88_val_gender, val_labels_gender)))
print ("model_88_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_88_val_age_group, val_labels_age_group)))
print ("model_88_males_original_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_88_males_original_val_age_group, val_labels_age_group_males)))
print ("model_88_females_original_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_88_females_original_val_age_group, val_labels_age_group_females)))

print ("model_53_val_age accuracy " + str(getAgeGroupModelAccuracy(model_53_val_age, val_labels_age_group)))
print ("model_53_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_53_val_age_group, val_labels_age_group)))
print ("model_31_val accuracy " + str(getAgeGroupModelAccuracy(model_31_val, val_labels_age_group)))
print ("model_48_val accuracy " + str(getAgeGroupModelAccuracy(model_48_val, val_labels_age_group)))
print ("model_43_val accuracy " + str(getAgeGroupModelAccuracy(model_43_val, val_labels_age_group)))
print ("model_43_males_original_val accuracy " + str(getAgeGroupModelAccuracy(model_43_males_original_val, val_labels_age_group_males)))
print ("model_43_females_original_val accuracy " + str(getAgeGroupModelAccuracy(model_43_females_original_val, val_labels_age_group_females)))
print ("model_43_males_val accuracy " + str(getAgeGroupModelAccuracy(model_43_males_val, val_labels_age_group_males)))
print ("model_43_females_val accuracy " + str(getAgeGroupModelAccuracy(model_43_females_val, val_labels_age_group_females)))

print ("model_84_val accuracy " + str(getAgeGroupModelAccuracy(model_84_val, val_labels_age_group)))
print ("model_85_val accuracy " + str(getAgeGroupModelAccuracy(model_85_val, val_labels_age_group)))
print ("model_26_val accuracy " + str(getAgeGroupModelAccuracy(model_26_val, val_labels_age_group)))
print ("model_51_val accuracy " + str(getAgeGroupModelAccuracy(model_51_val, val_labels_age_group)))
print ("model_54_val accuracy " + str(getAgeGroupModelAccuracy(model_54_val, val_labels_age_group)))
print ("model_91_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_91_val_age_group, val_labels_age_group)))
print ("model_91_val_gender accuracy " + str(getAgeGroupModelAccuracy(model_91_val_gender, val_labels_gender)))
print ("model_94_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_94_val_age_group, val_labels_age_group)))
print ("model_94_val_gender accuracy " + str(getAgeGroupModelAccuracy(model_94_val_gender, val_labels_gender)))
print ("model_94_males_original_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_94_males_original_val_age_group, val_labels_age_group_males)))
print ("model_94_females_original_val_age_group accuracy " + str(getAgeGroupModelAccuracy(model_94_females_original_val_age_group, val_labels_age_group_females)))




print ("\nTEST ACCURACIES")

print ("dnn_1_test accuracy " + str(getAgeGroupModelAccuracy(dnn_1_test, test_labels_age_group)))
print ("dnn_2_test accuracy " + str(getAgeGroupModelAccuracy(dnn_2_test, test_labels_age_group)))
print ("dnn_4_test accuracy " + str(getAgeGroupModelAccuracy(dnn_4_test, test_labels_age_group)))
print ("dnn_4_test_gender accuracy " + str(getAgeGroupModelAccuracy(dnn_4_test_gender, test_labels_gender)))
print ("dnn_4_males_original_test accuracy " + str(getAgeGroupModelAccuracy(dnn_4_males_original_test, test_labels_age_group_males)))
print ("dnn_4_females_original_test accuracy " + str(getAgeGroupModelAccuracy(dnn_4_females_original_test, test_labels_age_group_females)))

print ("model_77_test accuracy " + str(getAgeGroupModelAccuracy(model_77_test, test_labels_age_group)))
print ("model_87_test accuracy " + str(getAgeGroupModelAccuracy(model_87_test, test_labels_age_group)))
print ("model_87_test_gender accuracy " + str(getAgeGroupModelAccuracy(model_87_test_gender, test_labels_gender)))
print ("model_87_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_87_males_original_test, test_labels_age_group_males)))
print ("model_87_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_87_females_original_test, test_labels_age_group_females)))

print ("model_86_test accuracy " + str(getAgeGroupModelAccuracy(model_86_test, test_labels_age_group)))
print ("model_86_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_86_males_original_test, test_labels_age_group_males)))
print ("model_86_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_86_females_original_test, test_labels_age_group_females)))
print ("model_86_males_test accuracy " + str(getAgeGroupModelAccuracy(model_86_males_test, test_labels_age_group_males)))
print ("model_86_females_test accuracy " + str(getAgeGroupModelAccuracy(model_86_females_test, test_labels_age_group_females)))

print ("dnn_6_test_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_test_age, test_labels_age_group)))
print ("model_72_test accuracy " + str(getAgeGroupModelAccuracy(model_72_test, test_labels_age_group)))
print ("model_72_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_72_males_original_test, test_labels_age_group_males)))
print ("model_72_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_72_females_original_test, test_labels_age_group_females)))
print ("model_72_males_test accuracy " + str(getAgeGroupModelAccuracy(model_72_males_test, test_labels_age_group_males)))
print ("model_72_females_test accuracy " + str(getAgeGroupModelAccuracy(model_72_females_test, test_labels_age_group_females)))
print ("model_57_test accuracy " + str(getAgeGroupModelAccuracy(model_57_test, test_labels_age_group)))
print ("model_57_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_57_males_original_test, test_labels_age_group_males)))
print ("model_57_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_57_females_original_test, test_labels_age_group_females)))
print ("model_57_males_test accuracy " + str(getAgeGroupModelAccuracy(model_57_males_test, test_labels_age_group_males)))
print ("model_57_females_test accuracy " + str(getAgeGroupModelAccuracy(model_57_females_test, test_labels_age_group_females)))
print ("model_56_test accuracy " + str(getAgeGroupModelAccuracy(model_56_test, test_labels_age_group)))
print ("model_56_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_56_males_original_test, test_labels_age_group_males)))
print ("model_56_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_56_females_original_test, test_labels_age_group_females)))
print ("model_67_test accuracy " + str(getAgeGroupModelAccuracy(model_67_test, test_labels_age_group)))
print ("model_59_test accuracy " + str(getAgeGroupModelAccuracy(model_59_test, test_labels_age_group)))
print ("model_69_test accuracy " + str(getAgeGroupModelAccuracy(model_69_test, test_labels_age_group)))
print ("model_63_test accuracy " + str(getAgeGroupModelAccuracy(model_63_test, test_labels_age_group)))
print ("model_63_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_63_males_original_test, test_labels_age_group_males)))
print ("model_63_males_test accuracy " + str(getAgeGroupModelAccuracy(model_63_males_test, test_labels_age_group_males)))
print ("model_63_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_63_females_original_test, test_labels_age_group_females)))
print ("model_63_females_test accuracy " + str(getAgeGroupModelAccuracy(model_63_females_test, test_labels_age_group_females)))
print ("dnn_7_test_gender accuracy " + str(getAgeGroupModelAccuracy(dnn_7_test_gender, test_labels_gender)))
print ("dnn_7_test_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_7_test_age_group, test_labels_age_group)))
print ("dnn_7_males_original_age_group_test accuracy " + str(getAgeGroupModelAccuracy(dnn_7_males_original_age_group_test, test_labels_age_group_males)))
print ("dnn_7_females_original_age_group_test accuracy " + str(getAgeGroupModelAccuracy(dnn_7_females_original_age_group_test, test_labels_age_group_females)))
print ("model_23_test accuracy " + str(getAgeGroupModelAccuracy(model_23_test, test_labels_age_group)))
print ("model_38_test accuracy " + str(getAgeGroupModelAccuracy(model_38_test, test_labels_age_group)))
print ("model_61_test accuracy " + str(getAgeGroupModelAccuracy(model_61_test, test_labels_age_group)))
print ("dnn_3_test accuracy " + str(getAgeGroupModelAccuracy(dnn_3_test, test_labels_gender)))
print ("gender_1_test accuracy " + str(getAgeGroupModelAccuracy(gender_1_test, test_labels_gender)))
print ("gender_8_test accuracy " + str(getAgeGroupModelAccuracy(gender_8_test, test_labels_gender)))
print ("gender_4_test accuracy " + str(getAgeGroupModelAccuracy(gender_4_test, test_labels_gender)))
print ("gender_5_test accuracy " + str(getAgeGroupModelAccuracy(gender_5_test, test_labels_gender)))
print ("gender_6_test accuracy " + str(getAgeGroupModelAccuracy(gender_6_test, test_labels_gender)))
print ("gender_2_test accuracy " + str(getAgeGroupModelAccuracy(gender_2_test, test_labels_gender)))
print ("gender_9_test accuracy " + str(getAgeGroupModelAccuracy(gender_9_test, test_labels_gender)))
print ("model_90_test accuracy " + str(getAgeGroupModelAccuracy(model_90_test, test_labels_gender)))
print ("model_71_test accuracy " + str(getAgeGroupModelAccuracy(model_71_test, test_labels_age_group)))
print ("model_62_test accuracy " + str(getAgeGroupModelAccuracy(model_62_test, test_labels_age_group)))
print ("model_68_test accuracy " + str(getAgeGroupModelAccuracy(model_68_test, test_labels_age_group)))
print ("model_68_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_68_males_original_test, test_labels_age_group_males)))
print ("model_68_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_68_females_original_test, test_labels_age_group_females)))
print ("model_68_males_test accuracy " + str(getAgeGroupModelAccuracy(model_68_males_test, test_labels_age_group_males)))
print ("model_68_females_test accuracy " + str(getAgeGroupModelAccuracy(model_68_females_test, test_labels_age_group_females)))


print ("model_58_test accuracy " + str(getAgeGroupModelAccuracy(model_58_test, test_labels_age_group)))
print ("model_65_test accuracy " + str(getAgeGroupModelAccuracy(model_65_test, test_labels_age_group)))
print ("model_65_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_65_males_original_test, test_labels_age_group_males)))
print ("model_65_males_test accuracy " + str(getAgeGroupModelAccuracy(model_65_males_test, test_labels_age_group_males)))
print ("model_65_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_65_females_original_test, test_labels_age_group_females)))
print ("model_65_females_test accuracy " + str(getAgeGroupModelAccuracy(model_65_females_test, test_labels_age_group_females)))
print ("model_58_test accuracy " + str(getAgeGroupModelAccuracy(model_58_test, test_labels_age_group)))
print ("model_41_test accuracy " + str(getAgeGroupModelAccuracy(model_41_test, test_labels_age_group)))
print ("model_39_test accuracy " + str(getAgeGroupModelAccuracy(model_39_test, test_labels_age_group)))
print ("model_75_test accuracy " + str(getAgeGroupModelAccuracy(model_75_test, test_labels_age_group)))
print ("model_75_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_75_males_original_test, test_labels_age_group_males)))
print ("model_75_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_75_females_original_test, test_labels_age_group_females)))
print ("model_64_test accuracy " + str(getAgeGroupModelAccuracy(model_64_test, test_labels_age_group)))
print ("dnn_2_males_original_test accuracy " + str(getAgeGroupModelAccuracy(dnn_2_males_original_test, test_labels_age_group_males)))
print ("dnn_2_males_test accuracy " + str(getAgeGroupModelAccuracy(dnn_2_males_test, test_labels_age_group_males)))
print ("dnn_2_females_original_test accuracy " + str(getAgeGroupModelAccuracy(dnn_2_females_original_test, test_labels_age_group_females)))
print ("dnn_2_females_test accuracy " + str(getAgeGroupModelAccuracy(dnn_2_females_test, test_labels_age_group_females)))
print ("dnn_1_males_loss_test accuracy " + str(getAgeGroupModelAccuracy(dnn_1_males_loss_test, test_labels_age_group_males)))
print ("dnn_1_males_original_test accuracy " + str(getAgeGroupModelAccuracy(dnn_1_males_original_test, test_labels_age_group_males)))
print ("dnn_1_males_test accuracy " + str(getAgeGroupModelAccuracy(dnn_1_males_test, test_labels_age_group_males)))
print ("dnn_1_females_original_test accuracy " + str(getAgeGroupModelAccuracy(dnn_1_females_original_test, test_labels_age_group_females)))
print ("dnn_1_females_test accuracy " + str(getAgeGroupModelAccuracy(dnn_1_females_test, test_labels_age_group_females)))
print ("model_42_test accuracy " + str(getAgeGroupModelAccuracy(model_42_test, test_labels_age_group)))
print ("model_42_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_42_males_original_test, test_labels_age_group_males)))
print ("model_42_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_42_females_original_test, test_labels_age_group_females)))
print ("model_66_test accuracy " + str(getAgeGroupModelAccuracy(model_66_test, test_labels_age_group)))
print ("model_66_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_66_males_original_test, test_labels_age_group_males)))
print ("model_66_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_66_females_original_test, test_labels_age_group_females)))
print ("model_37_test accuracy " + str(getAgeGroupModelAccuracy(model_37_test, test_labels_age_group)))
print ("model_93_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_93_test_age_group, test_labels_age_group)))
print ("model_93_test_gender accuracy " + str(getAgeGroupModelAccuracy(model_93_test_gender, test_labels_gender)))

print ("model_95_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_95_test_age_group, test_labels_age_group)))
print ("model_95_test_gender accuracy " + str(getAgeGroupModelAccuracy(model_95_test_gender, test_labels_gender)))
#print ("model_91_test_gender accuracy " + str(getAgeGroupModelAccuracy(model_91_test_gender, test_labels_gender)))
print ("model_88_test_gender accuracy " + str(getAgeGroupModelAccuracy(model_88_test_gender, test_labels_gender)))
print ("model_88_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_88_test_age_group, test_labels_age_group)))
print ("model_88_males_original_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_88_males_original_test_age_group, test_labels_age_group_males)))
print ("model_88_females_original_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_88_females_original_test_age_group, test_labels_age_group_females)))


print ("model_53_test_age accuracy " + str(getAgeGroupModelAccuracy(model_53_test_age, test_labels_age_group)))
print ("model_53_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_53_test_age_group, test_labels_age_group)))
print ("model_31_test accuracy " + str(getAgeGroupModelAccuracy(model_31_test, test_labels_age_group)))
print ("model_48_test accuracy " + str(getAgeGroupModelAccuracy(model_48_test, test_labels_age_group)))
print ("model_43_test accuracy " + str(getAgeGroupModelAccuracy(model_43_test, test_labels_age_group)))
print ("model_43_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_43_males_original_test, test_labels_age_group_males)))
print ("model_43_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_43_females_original_test, test_labels_age_group_females)))
print ("model_43_males_test accuracy " + str(getAgeGroupModelAccuracy(model_43_males_test, test_labels_age_group_males)))
print ("model_43_females_test accuracy " + str(getAgeGroupModelAccuracy(model_43_females_test, test_labels_age_group_females)))
print ("model_84_test accuracy " + str(getAgeGroupModelAccuracy(model_84_test, test_labels_age_group)))
print ("model_85_test accuracy " + str(getAgeGroupModelAccuracy(model_85_test, test_labels_age_group)))
print ("model_26_test accuracy " + str(getAgeGroupModelAccuracy(model_26_test, test_labels_age_group)))
print ("model_82_test accuracy " + str(getAgeGroupModelAccuracy(model_82_test, test_labels_age_group)))
print ("model_51_test accuracy " + str(getAgeGroupModelAccuracy(model_51_test, test_labels_age_group)))
print ("model_54_test accuracy " + str(getAgeGroupModelAccuracy(model_54_test, test_labels_age_group)))
print ("model_91_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_91_test_age_group, test_labels_age_group)))
print ("model_91_test_gender accuracy " + str(getAgeGroupModelAccuracy(model_91_test_gender, test_labels_gender)))
print ("model_94_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_94_test_age_group, test_labels_age_group)))
print ("model_94_test_gender accuracy " + str(getAgeGroupModelAccuracy(model_94_test_gender, test_labels_gender)))
print ("model_94_males_original_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_94_males_original_test_age_group, test_labels_age_group_males)))
print ("model_94_females_original_test_age_group accuracy " + str(getAgeGroupModelAccuracy(model_94_females_original_test_age_group, test_labels_age_group_females)))



'''
print ("combined_predictions_val accuracy " + str(getAgeGroupModelAccuracy(combined_predictions_val, val_labels_age_group)))
print ("combined_predictions_weighted_val accuracy " + str(getAgeGroupModelAccuracy(combined_predictions_weighted_val, val_labels_age_group)))
print ("combined_predictions_test accuracy " + str(getAgeGroupModelAccuracy(combined_predictions_test, test_labels_age_group)))
print ("combined_predictions_weighted_test accuracy " + str(getAgeGroupModelAccuracy(combined_predictions_weighted_test, test_labels_age_group)))


print ("combined_predictions_val_2 accuracy " + str(getAgeGroupModelAccuracy(combined_predictions_val_2, val_labels_age_group)))
print ("combined_predictions_test_2 accuracy " + str(getAgeGroupModelAccuracy(combined_predictions_test_2, test_labels_age_group)))
print ("combined_predictions_weighted_val_2 accuracy " + str(getAgeGroupModelAccuracy(combined_predictions_weighted_val_2, val_labels_age_group)))
print ("combined_predictions_weighted_test_2 accuracy " + str(getAgeGroupModelAccuracy(combined_predictions_weighted_test_2, test_labels_age_group)))
'''




print ("model_68_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_68_males_original_test, test_labels_age_group_males)))
print ("model_68_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_68_females_original_test, test_labels_age_group_females)))
print ("model_67_males_original_test accuracy " + str(getAgeGroupModelAccuracy(model_67_males_original_test, test_labels_age_group_males)))
print ("model_67_females_original_test accuracy " + str(getAgeGroupModelAccuracy(model_67_females_original_test, test_labels_age_group_females)))


print ("dnn_6_males_original_test_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_6_males_original_test_age_group, test_labels_age_group_males)))
print ("dnn_6_females_original_test_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_6_females_original_val_age_group, test_labels_age_group_females)))
print ("dnn_6_males_original_test_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_males_original_test_age, test_labels_age_group_males)))
print ("dnn_6_females_original_test_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_females_original_test_age, test_labels_age_group_females)))

print ("dnn_6_males_test_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_6_males_test_age_group, test_labels_age_group_males)))
print ("dnn_6_females_test_age_group accuracy " + str(getAgeGroupModelAccuracy(dnn_6_females_test_age_group, test_labels_age_group_females)))
print ("dnn_6_males_test_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_males_test_age, test_labels_age_group_males)))
print ("dnn_6_females_test_age accuracy " + str(getAgeGroupModelAccuracy(dnn_6_females_test_age, test_labels_age_group_females)))

'''
#AVERAGING
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



#MAJORITY VOTING
def majorityVoting(predictions, accuracies, data):
    correct_predictions = []
    for j in range(0, len(predictions[0])):
        predicted = []
        for i in range(0, len(predictions)):
            predicted.append(np.argmax(predictions[i][j]))
        c = Counter(predicted)
        commonElements = c.most_common(3)
        # CHECK IF ONE ELEMENT WAS MORE POPULAR THAN OTHERS
        prediction = -1
        if (len(commonElements) > 1):
            if (commonElements[0][1] == commonElements[1][1]):
                prediction = np.argmax(predictions[np.argmax(accuracies)][j])
            else:
                prediction = commonElements[0][0]
        else:
            prediction = commonElements[0][0]

        #CHECK IF MERGE PREDICTION WAS CORRECT
        if (prediction == np.argmax(data[j])):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    return correct_predictions.count(1)/len(correct_predictions)



print ("\n\n1) MAJORITY VOTING AND AVERAGING FOR MALES' AGE GROUP IDENTIFICATION")

# LOAD MODEL ACCURACIES
model_43_males_val_acc = getAgeGroupModelAccuracy(model_43_males_val, val_labels_age_group_males)
dnn_2_males_val_acc = getAgeGroupModelAccuracy(dnn_2_males_val, val_labels_age_group_males)
model_57_males_val_acc = getAgeGroupModelAccuracy(model_57_males_val, val_labels_age_group_males)
model_65_males_val_acc = getAgeGroupModelAccuracy(model_65_males_val, val_labels_age_group_males)
model_88_males_original_val_acc = getAgeGroupModelAccuracy(model_88_males_original_val_age_group, val_labels_age_group_males)

model_43_males_test_acc = getAgeGroupModelAccuracy(model_43_males_test, test_labels_age_group_males)
dnn_2_males_test_acc = getAgeGroupModelAccuracy(dnn_2_males_test, test_labels_age_group_males)
model_57_males_test_acc = getAgeGroupModelAccuracy(model_57_males_test, test_labels_age_group_males)
model_65_males_test_acc = getAgeGroupModelAccuracy(model_65_males_test, test_labels_age_group_males)
model_88_males_original_test_acc = getAgeGroupModelAccuracy(model_88_males_original_test_age_group, test_labels_age_group_males)

print ("======MAJORITY VOTING======")
print ("1) MALES ON VALIDATION DATA " + str(majorityVoting(
            [model_43_males_val, dnn_2_males_val, model_57_males_val, model_65_males_val],
            [model_43_males_val_acc, dnn_2_males_val_acc, model_57_males_val_acc, model_65_males_val_acc],
            val_labels_age_group_males)))
print ("1) MALES ON TEST DATA " + str(majorityVoting(
            [model_43_males_test, dnn_2_males_test, model_57_males_test, model_65_males_test],
            [model_43_males_test_acc, dnn_2_males_test_acc, model_57_males_test_acc, model_65_males_test_acc],
            test_labels_age_group_males)))

print ("======AVERAGING======")
print ("1) MALES ON VALIDATION DATA " + str(averagePredictions(
            [model_43_males_val, dnn_2_males_val, model_57_males_val, model_65_males_val],
            [0.3, 0.3, 0.3, 0.2],
            val_labels_age_group_males)))
print ("1) MALES ON TEST DATA " + str(averagePredictions(
            [model_43_males_test, dnn_2_males_test, model_57_males_test, model_65_males_test],
            [0.3, 0.3, 0.3, 0.2],
            test_labels_age_group_males)))



print ("\n\n2) MAJORITY VOTING AND AVERAGING FOR FEMALES' AGE GROUP IDENTIFICATION")

# LOAD MODEL ACCURACIES
model_63_females_original_val_acc = getAgeGroupModelAccuracy(model_63_females_original_val, val_labels_age_group_females)
dnn_6_females_original_val_age_acc = getAgeGroupModelAccuracy(dnn_6_females_original_val_age, val_labels_age_group_females)
dnn_7_females_original_age_group_val_acc = getAgeGroupModelAccuracy(dnn_7_females_original_age_group_val, val_labels_age_group_females)
model_94_females_original_val_age_group_acc = getAgeGroupModelAccuracy(model_94_females_original_val_age_group, val_labels_age_group_females)
model_65_females_original_val_acc = getAgeGroupModelAccuracy(model_65_females_original_val, val_labels_age_group_females)

model_63_females_original_test_acc = getAgeGroupModelAccuracy(model_63_females_original_test, test_labels_age_group_females)
dnn_6_females_original_test_age_acc = getAgeGroupModelAccuracy(dnn_6_females_original_test_age, test_labels_age_group_females)
dnn_7_females_original_age_group_test_acc = getAgeGroupModelAccuracy(dnn_7_females_original_age_group_test, test_labels_age_group_females)
model_94_females_original_test_age_group_acc = getAgeGroupModelAccuracy(model_94_females_original_test_age_group, test_labels_age_group_females)
model_65_females_original_test_acc = getAgeGroupModelAccuracy(model_65_females_original_test, test_labels_age_group_females)


print ("======MAJORITY VOTING======")
print ("FEMALES ON VALIDATION DATA " + str(majorityVoting(
            [model_63_females_original_val, dnn_6_females_original_val_age, model_94_females_original_val_age_group, model_65_females_original_val],
            [model_63_females_original_val_acc, dnn_6_females_original_val_age_acc, model_94_females_original_val_age_group_acc, model_65_females_original_val_acc],
            val_labels_age_group_females)))
print ("FEMALES ON TEST DATA " + str(majorityVoting(
            [model_63_females_original_test, dnn_6_females_original_test_age, model_94_females_original_test_age_group, model_65_females_original_test],
            [model_63_females_original_test_acc, dnn_6_females_original_test_age_acc,  model_94_females_original_test_age_group_acc, model_65_females_original_test_acc],
            test_labels_age_group_females)))

print ("======AVERAGING======")
print ("FEMALES ON VALIDATION DATA " + str(averagePredictions(
            [model_63_females_original_val, dnn_6_females_original_val_age, model_94_females_original_val_age_group, model_65_females_original_val],
            [0.6, 0.3, 0.2, 0.3],
            val_labels_age_group_females)))
print ("FEMALES ON TEST DATA " + str(averagePredictions(
            [model_63_females_original_test, dnn_6_females_original_test_age, model_94_females_original_test_age_group, model_65_females_original_test],
            [0.6, 0.3, 0.2, 0.3],
            test_labels_age_group_females)))




print ("\n\n3) MAJORITY VOTING AND AVERAGING FOR GENDER IDENTIFICATION")

# LOAD MODEL ACCURACIES
dnn_3_val_acc = getAgeGroupModelAccuracy(dnn_3_val, val_labels_gender)
dnn_7_val_gender_acc = getAgeGroupModelAccuracy(dnn_7_val_gender, val_labels_gender)
model_90_val_acc = getAgeGroupModelAccuracy(model_90_val, val_labels_gender)
dnn_4_val_gender_acc = getAgeGroupModelAccuracy(dnn_4_val_gender, val_labels_gender)
model_87_val_acc = getAgeGroupModelAccuracy(model_87_val, val_labels_gender)

dnn_3_test_acc = getAgeGroupModelAccuracy(dnn_3_test, test_labels_gender)
dnn_7_test_gender_acc = getAgeGroupModelAccuracy(dnn_7_test_gender, test_labels_gender)
model_90_test_acc = getAgeGroupModelAccuracy(model_90_test, test_labels_gender)
dnn_4_test_gender_acc = getAgeGroupModelAccuracy(dnn_4_test_gender, test_labels_gender)
model_87_test_gender_acc = getAgeGroupModelAccuracy(model_87_test_gender, test_labels_gender)


print ("======MAJORITY VOTING======")
print ("GENDER ON VALIDATION DATA " + str(majorityVoting(
            [dnn_3_val, dnn_7_val_gender, model_90_val, dnn_4_val_gender],
            [dnn_3_val_acc, dnn_7_val_gender_acc, model_90_val_acc, dnn_4_val_gender_acc],
            val_labels_gender)))
print ("GENDER ON TEST DATA " + str(majorityVoting(
            [dnn_3_test, dnn_7_test_gender, model_90_test, dnn_4_test_gender, model_87_test_gender],
            [dnn_3_test_acc, dnn_7_test_gender_acc, model_90_test_acc, dnn_4_test_gender_acc, model_87_test_gender_acc],
            test_labels_gender)))
print ("======AVERAGING======")
print ("GENDER ON VALIDATION DATA " + str(averagePredictions(
            [dnn_3_val, dnn_7_val_gender, model_90_val, dnn_4_val_gender],
            [0.2, 0.3, 0.3, 0.2, 0.1],
            val_labels_gender)))
print ("GENDER ON TEST DATA " + str(averagePredictions(
            [dnn_3_test, dnn_7_test_gender, model_90_test, dnn_4_test_gender, model_87_test_gender],
            [0.2, 0.3, 0.3, 0.2, 0.1],
            test_labels_gender)))



print ("\n\n4) MAJORITY VOTING AND AVERAGING FOR AGE GROUP IDENTIFICATION")

# LOAD MODEL ACCURACIES
model_63_val_acc = getAgeGroupModelAccuracy(model_63_val, val_labels_age_group)
dnn_7_val_age_group_acc = getAgeGroupModelAccuracy(dnn_7_val_age_group, val_labels_age_group)
dnn_2_val_acc = getAgeGroupModelAccuracy(dnn_2_val, val_labels_gender)
model_68_val_acc = getAgeGroupModelAccuracy(model_68_val, val_labels_age_group)
dnn_6_val_age_acc =  getAgeGroupModelAccuracy(dnn_6_val_age, val_labels_age_group)

model_63_test_acc = getAgeGroupModelAccuracy(model_63_test, test_labels_age_group)
dnn_7_test_age_group_acc = getAgeGroupModelAccuracy(dnn_7_test_age_group, test_labels_age_group)
dnn_2_test_acc = getAgeGroupModelAccuracy(dnn_2_test, test_labels_gender)
model_68_test_acc = getAgeGroupModelAccuracy(model_68_test, test_labels_age_group)
dnn_6_test_age_acc =  getAgeGroupModelAccuracy(dnn_6_test_age, test_labels_age_group)


print ("======MAJORITY VOTING======")
print ("AGE GROUP ON VALIDATION DATA " + str(majorityVoting(
            [model_63_val, dnn_7_val_age_group, model_68_val, dnn_6_val_age],
            [model_63_val_acc, dnn_7_val_age_group_acc, model_68_val_acc, dnn_6_val_age_acc],
            val_labels_age_group)))
print ("AGE GROUP ON TEST DATA " + str(majorityVoting(
            [model_63_test, dnn_7_test_age_group, dnn_2_test, dnn_6_test_age],
            [model_63_test_acc, dnn_7_test_age_group_acc, dnn_2_test_acc, dnn_6_test_age_acc],
            test_labels_age_group)))
print ("======AVERAGING======")
print ("AGE GROUP ON VALIDATION DATA " + str(averagePredictions(
            [model_63_val, dnn_7_val_age_group, dnn_2_val, dnn_6_val_age],
            [0.5, 0.5, 0.3, 0.1],
            val_labels_age_group)))
print ("AGE GROUP ON TEST DATA " + str(averagePredictions(
            [model_63_test, dnn_7_test_age_group, dnn_2_test, dnn_6_test_age],
            [0.5, 0.5, 0.3, 0.1],
            test_labels_age_group)))

