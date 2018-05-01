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


survey_results = pandas.read_csv("./survey.csv", sep=";")
survey_results = survey_results.rename(columns={'\ufeffpredictedAge': 'predictedAge'})
survey_results_predictions_over_40 = survey_results['predictedAge'][survey_results['age'] < 40]
survey_results_real_over_40 = survey_results['realAge'][survey_results['age'] < 40]


test_data_i_vectors = pandas.read_csv("/storage/tanel/child_age_gender/exp/ivectors_2048/test/export.csv", sep=" ")
test_data_i_vectors = format(test_data_i_vectors)
test_labels_age = test_data_i_vectors['Age']
test_labels_age_group = onehot(test_data_i_vectors['AgeGroup'])
test_labels_age_group_males = onehot(test_data_i_vectors['AgeGroup'][test_data_i_vectors['Gender'] == 0])
test_labels_age_group_females = onehot(test_data_i_vectors['AgeGroup'][test_data_i_vectors['Gender'] == 1])

test_data_indexes_used_in_survey = []
test_data_males_indexes_used_in_survey = []
test_data_females_indexes_used_in_survey = []





for i in range(0, len(test_data_i_vectors)):
    not_in_excluded = []
    for j in range(1, 10):
        if ('_' + str(j) + '_' not in test_data_i_vectors.iloc[i]['Utterance']):
            not_in_excluded.append(j)
    if (len(not_in_excluded) == 9):
        test_data_indexes_used_in_survey.append(i)
        if(test_data_i_vectors.iloc[i]['Gender'] == 0):
            test_data_males_indexes_used_in_survey.append(i)
        else:
            test_data_females_indexes_used_in_survey.append(i)


test_data_i_vectors_used_in_survey = test_data_i_vectors.iloc[test_data_indexes_used_in_survey]
test_labels_gender = onehot(test_data_i_vectors['Gender'])
test_labels_age_group_used_in_survey_males = onehot(test_data_i_vectors_used_in_survey['AgeGroup'][test_data_i_vectors_used_in_survey['Gender'] == 0])
test_labels_age_group_used_in_survey_females = onehot(test_data_i_vectors_used_in_survey['AgeGroup'][test_data_i_vectors_used_in_survey['Gender'] == 1])
test_labels_age_group_used_in_survey = test_labels_age_group[test_data_indexes_used_in_survey]
test_labels_age_used_in_survey = test_labels_age[test_data_indexes_used_in_survey]
test_labels_age_used_in_survey = np.array(test_labels_age_used_in_survey.tolist())
test_labels_gender_used_in_survey = test_labels_gender[test_data_indexes_used_in_survey]


print ("LABELS LOADED")

def getAgeGroupModelAccuracy(predictions, data):
    correct_predictions = []
    for i in range(0, len(predictions)):
        if (np.argmax(predictions[i]) == np.argmax(data[i])):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    return correct_predictions.count(1)/len(predictions)

def getGenderModelAccuracy(predictions, real):
    correct_predictions = []
    predictions = predictions.tolist()
    real = real.tolist()

    for i in range(0, len(predictions)):
        if (predictions[i] == real[i]):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)
    return correct_predictions.count(1)/len(predictions)

def formatAgeDataToAgeGroups(data):
    predictions = []
    data = data.tolist()
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

def getSpecificAgeGroupDataIndexes(data, ageGroup):
    dataToReturn = []
    for i in range(0, len(data)):
        if (data.iloc[i]['realAge'] < 13 and ageGroup == 1):
            dataToReturn.append(i)
        elif (data.iloc[i]['realAge'] > 14 and ageGroup == 3):
            dataToReturn.append(i)
        elif (data.iloc[i]['realAge'] < 15 and data.iloc[i]['realAge'] > 12 and ageGroup == 2):
           dataToReturn.append(i)
    return dataToReturn

def getSpecificAgeGroupDataIndexes2(data, ageGroup):
    dataToReturn = []
    for i in range(0, len(data)):
        if (data[i] < 13 and ageGroup == 1):
            dataToReturn.append(i)
        elif (data[i] > 14 and ageGroup == 3):
            dataToReturn.append(i)
        elif (data[i] > 12 and data[i] < 15 and ageGroup == 2):
           dataToReturn.append(i)
    return dataToReturn

combined_prediction_1 = np.load("./test/combined_prediction.npy", encoding="bytes")
combined_prediction_2 = np.load("./test/combined_prediction_2.npy", encoding="bytes")
model_63 = np.load("./test/model_63.npy", encoding="bytes")
dnn_3 = np.load("./test/dnn_3_gender.npy", encoding="bytes")[test_data_indexes_used_in_survey]
dnn_2_males = np.load("./test/dnn_2_males.npy", encoding="bytes")
dnn_7_predictions = np.load("./test/dnn_7_age_group.npy", encoding="bytes")[test_data_indexes_used_in_survey]

print ("AGE GROUP")
print ("=======================")
print ("combined_prediction_1 accuracy on survey data: " + str(getAgeGroupModelAccuracy(combined_prediction_1[test_data_indexes_used_in_survey], test_labels_age_group_used_in_survey)))
print ("combined_prediction_2 accuracy on survey data: " + str(getAgeGroupModelAccuracy(combined_prediction_2[test_data_indexes_used_in_survey], test_labels_age_group_used_in_survey)))
print ("model_63 accuracy on survey data: " + str(getAgeGroupModelAccuracy(model_63[test_data_indexes_used_in_survey], test_labels_age_group_used_in_survey)))
print ("=======================")
print ("MALES AGE GROUP")
print ("=======================")
print ("combined_prediction_1 accuracy on survey data: " + str(getAgeGroupModelAccuracy(combined_prediction_1[test_data_males_indexes_used_in_survey], test_labels_age_group_used_in_survey_males)))
print ("combined_prediction_2 accuracy on survey data: " + str(getAgeGroupModelAccuracy(combined_prediction_2[test_data_males_indexes_used_in_survey], test_labels_age_group_used_in_survey_males)))
print ("model_63 accuracy on survey data: " + str(getAgeGroupModelAccuracy(model_63[test_data_males_indexes_used_in_survey], test_labels_age_group_used_in_survey_males)))
print ("=======================")
print ("FEMALES AGE GROUP")
print ("=======================")
print ("combined_prediction_1 accuracy on survey data: " + str(getAgeGroupModelAccuracy(combined_prediction_1[test_data_females_indexes_used_in_survey], test_labels_age_group_used_in_survey_females)))
print ("combined_prediction_2 accuracy on survey data: " + str(getAgeGroupModelAccuracy(combined_prediction_2[test_data_females_indexes_used_in_survey], test_labels_age_group_used_in_survey_females)))
print ("model_63 accuracy on survey data: " + str(getAgeGroupModelAccuracy(model_63[test_data_females_indexes_used_in_survey], test_labels_age_group_used_in_survey_females)))
print ("=======================")
print ("GENDER")
print ("=======================")
print ("dnn_3 accuracy on survey data: " + str(getAgeGroupModelAccuracy(dnn_3, test_labels_gender_used_in_survey)))
print ("=======================")
print ("\n\n\nSURVEY ANALYSIS")
print ("=======================")
print ("PARTICIPANTS")
print ("=======================")
print ("Total number of participants ", survey_results.Date.nunique())
print ("Total number of males ", survey_results[survey_results['gender'] == 'Male'].Date.nunique())
print ("Total number of females ", survey_results[survey_results['gender'] == 'Female'].Date.nunique())
print ("Total number of Estonians ", survey_results[survey_results['language'] == 'Estonian'].Date.nunique())
print ("Total number of Other ", survey_results[survey_results['language'] == 'Other'].Date.nunique())
print ("=======================")
print ("OVERALL ACCURACIES")
print ("=======================")
print ("Overall accuracy of age group identificaiton ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results['predictedAge']), formatAgeDataToAgeGroups(survey_results['realAge'])))
print ("Overall accuracy of gender identificaiton ", getGenderModelAccuracy(survey_results['predictedGender'], survey_results['realGender']))
print ("Accuracy of age group identificaiton of male speakers ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['realGender'] == 'Male']['predictedAge'].values), formatAgeDataToAgeGroups(survey_results[survey_results['realGender'] == 'Male']['realAge'].values)))
print ("Accuracy of age group identificaiton of male speakers ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['realGender'] == 'Female']['predictedAge'].values), formatAgeDataToAgeGroups(survey_results[survey_results['realGender'] == 'Female']['realAge'].values)))
print ("=======================")
print ("MALE PARTICIPANTS ACCURACIES")
print ("=======================")
print ("Males accuracy of age group identificaiton ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['gender'] == 'Male']['predictedAge'].values), formatAgeDataToAgeGroups(survey_results[survey_results['gender'] == 'Male']['realAge'].values)))
print ("Males accuracy of gender identificaiton ", getGenderModelAccuracy(survey_results[survey_results['gender'] == 'Male']['predictedGender'].values, survey_results[survey_results['gender'] == 'Male']['realGender'].values))
print ("Males accuracy of age group identificaiton of male speakers ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[(survey_results['gender'] == 'Male') & (survey_results['realGender'] == 'Male')]['predictedAge'].values), formatAgeDataToAgeGroups(survey_results[(survey_results['gender'] == 'Male') & (survey_results['realGender'] == 'Male')]['realAge'].values)))
print ("Males accuracy of age group identificaiton of male speakers ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[(survey_results['gender'] == 'Male') & (survey_results['realGender'] == 'Female')]['predictedAge'].values), formatAgeDataToAgeGroups(survey_results[(survey_results['gender'] == 'Male') & (survey_results['realGender'] == 'Female')]['realAge'].values)))
print ("=======================")
print ("FEMALE PARTICIPANTS ACCURACIES")
print ("=======================")
print ("Females accuracy of age group identificaiton ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['gender'] == 'Female']['predictedAge'].values), formatAgeDataToAgeGroups(survey_results[survey_results['gender'] == 'Female']['realAge'].values)))
print ("Females accuracy of gender identificaiton ", getGenderModelAccuracy(survey_results[survey_results['gender'] == 'Female']['predictedGender'].values, survey_results[survey_results['gender'] == 'Female']['realGender'].values))
print ("Females accuracy of age group identificaiton of male speakers ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[(survey_results['gender'] == 'Female') & (survey_results['realGender'] == 'Male')]['predictedAge'].values), formatAgeDataToAgeGroups(survey_results[(survey_results['gender'] == 'Female') & (survey_results['realGender'] == 'Male')]['realAge'].values)))
print ("Females accuracy of age group identificaiton of male speakers ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[(survey_results['gender'] == 'Female') & (survey_results['realGender'] == 'Female')]['predictedAge'].values), formatAgeDataToAgeGroups(survey_results[(survey_results['gender'] == 'Female') & (survey_results['realGender'] == 'Female')]['realAge'].values)))
print ("=======================")
print ("EXPERIMENTS FOR AGE GROUP IDENTIFICATION")
print ("=======================")
print ("Overall accuracy for over 40 ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['age'] >= 40]['predictedAge']), formatAgeDataToAgeGroups(survey_results[survey_results['age'] >= 40]['realAge'])))
print ("Overall accuracy for under 40 ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['age'] < 40]['predictedAge']), formatAgeDataToAgeGroups(survey_results[survey_results['age'] < 40]['realAge'])))
print ("Overall accuracy for Estonians ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['language'] == 'Estonian']['predictedAge']), formatAgeDataToAgeGroups(survey_results[survey_results['language'] == 'Estonian']['realAge'])))
print ("Overall accuracy for non-Estonians ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['language'] != 'Estonian']['predictedAge']), formatAgeDataToAgeGroups(survey_results[survey_results['language'] != 'Estonian']['realAge'])))
print ("Overall accuracy for participants who have children ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['nrOfChildren'] > 0]['predictedAge']), formatAgeDataToAgeGroups(survey_results[survey_results['nrOfChildren'] > 0]['realAge'])))
print ("Overall accuracy for participants who do not have children ", getAgeGroupModelAccuracy(formatAgeDataToAgeGroups(survey_results[survey_results['nrOfChildren'] == 0]['predictedAge']), formatAgeDataToAgeGroups(survey_results[survey_results['nrOfChildren'] ==  0]['realAge'])))
print ("=======================")
print ("EXPERIMENTS FOR GENDER IDENTIFICATION")
print ("=======================")
print ("Overall accuracy for over 40 ", getGenderModelAccuracy((survey_results[survey_results['age'] >= 40]['predictedGender']), (survey_results[survey_results['age'] >= 40]['realGender'])))
print ("Overall accuracy for under 40 ", getGenderModelAccuracy((survey_results[survey_results['age'] < 40]['predictedGender']), (survey_results[survey_results['age'] < 40]['realGender'])))
print ("Overall accuracy for Estonians ", getGenderModelAccuracy((survey_results[survey_results['language'] == 'Estonian']['predictedGender']), (survey_results[survey_results['language'] == 'Estonian']['realGender'])))
print ("Overall accuracy for non-Estonians ", getGenderModelAccuracy((survey_results[survey_results['language'] != 'Estonian']['predictedGender']), (survey_results[survey_results['language'] != 'Estonian']['realGender'])))
print ("Overall accuracy for participants who have children ", getGenderModelAccuracy((survey_results[survey_results['nrOfChildren'] > 0]['predictedGender']), (survey_results[survey_results['nrOfChildren'] > 0]['realGender'])))
print ("Overall accuracy for participants who do not have children ", getGenderModelAccuracy((survey_results[survey_results['nrOfChildren'] == 0]['predictedGender']), (survey_results[survey_results['nrOfChildren'] ==  0]['realGender'])))
print ("=======================")
print ("MEAN ERRORS")
print ("=======================")
print ("Mean absoulte error", mean_absolute_error(survey_results['predictedAge'].values, survey_results['realAge'].values))
print ("Mean absolute error for first age group", mean_absolute_error(survey_results.iloc[getSpecificAgeGroupDataIndexes(survey_results, 1)]['predictedAge'], (survey_results.iloc[getSpecificAgeGroupDataIndexes(survey_results, 1)]['realAge'])))
print ("Mean absolute error for second age group", mean_absolute_error(survey_results.iloc[getSpecificAgeGroupDataIndexes(survey_results, 2)]['predictedAge'], (survey_results.iloc[getSpecificAgeGroupDataIndexes(survey_results, 2)]['realAge'])))
print ("Mean absolute error for third age group", mean_absolute_error(survey_results.iloc[getSpecificAgeGroupDataIndexes(survey_results, 3)]['predictedAge'], (survey_results.iloc[getSpecificAgeGroupDataIndexes(survey_results, 3)]['realAge'])))
print ("Overall absoulte error of the model", mean_absolute_error(dnn_7_predictions, test_labels_age_used_in_survey))
print ("Mean absoulte error of the model for first age group", mean_absolute_error(dnn_7_predictions[getSpecificAgeGroupDataIndexes2(test_labels_age_used_in_survey, 1)], test_labels_age_used_in_survey[getSpecificAgeGroupDataIndexes2(test_labels_age_used_in_survey, 1)]))
print ("Mean absoulte error of the model for second age group", mean_absolute_error(dnn_7_predictions[getSpecificAgeGroupDataIndexes2(test_labels_age_used_in_survey, 2)], test_labels_age_used_in_survey[getSpecificAgeGroupDataIndexes2(test_labels_age_used_in_survey, 2)]))
print ("Mean absoulte error of the model for third age group", mean_absolute_error(dnn_7_predictions[getSpecificAgeGroupDataIndexes2(test_labels_age_used_in_survey, 3)], test_labels_age_used_in_survey[getSpecificAgeGroupDataIndexes2(test_labels_age_used_in_survey, 3)]))
