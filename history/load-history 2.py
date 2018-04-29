#! /usr/bin/python2.7

import sys
import sys
import pandas
import numpy as np

################################################################################################
modelHistory = np.load('./history_model_58.npy').item()
'''
print ("HISTORY: ")
print (modelHistory)

print ("\n\nMIN VAL_LOSS age_output_loss" + str(min(modelHistory['age_output_loss'])))
print ("AVG VAL_LOSS age_output_loss" + str(np.mean(modelHistory['age_output_loss'])))

print ("\n\nMIN VAL_LOSS val_group_output_loss" + str(min(modelHistory['val_group_output_loss'])))
print ("AVG VAL_LOSS val_group_output_loss" + str(np.mean(modelHistory['val_group_output_loss'])))

print ("\n\nMAX VAL_ACC val_group_output_acc" + str(max(modelHistory['val_group_output_acc'])))
print ("AVG VAL_ACC val_group_output_acc" + str(np.mean(modelHistory['val_group_output_acc'])))

print ("\n\nMAX VAL_ACC val_age_output_age_group_accuracy " + str(max(modelHistory['val_age_output_age_group_accuracy'])))
print ("AVG VAL_ACC val_age_output_age_group_accuracy" + str(np.mean(modelHistory['val_age_output_age_group_accuracy'])))


'''
print (modelHistory)
print ("\n\nMIN VAL_LOSS " + str(min(modelHistory['val_loss'])))
print ("AVG VAL_LOSS" + str(np.mean(modelHistory['val_loss'])))

print ("MAX VAL_ACC " + str(max(modelHistory['val_acc'])))
print ("AVG VAL_ACC " + str(np.mean(modelHistory['val_acc'])))
'''
print ("\nMIN LOSS " + str(min(modelHistory['val_gender_loss'])))
print ("AVG LOSS " + str(np.mean(modelHistory['val_gender_loss'])))


print ("MAX ACC " + str(max(modelHistory['val_gender_acc'])))
print ("AVG ACC " + str(np.mean(modelHistory['val_gender_acc'])) + "\n")
'''
########################################
