#! /usr/bin/python2.7

import sys
import sys
import pandas
import numpy as np
################################################################################################
modelHistory = np.load('./gender/gender_9.npy').item()

print ("HISTORY: ")
print (modelHistory)

print ("\n\nMIN VAL_LOSS " + str(min(modelHistory['val_loss'])))
print ("AVG VAL_LOSS " + str(np.mean(modelHistory['val_loss'])))
print ("MAX VAL_ACC " + str(max(modelHistory['val_acc'])))
print ("AVG VAL_ACC " + str(np.mean(modelHistory['val_acc'])))


print ("\nMIN LOSS " + str(min(modelHistory['loss'])))
print ("AVG LOSS " + str(np.mean(modelHistory['loss'])))
print ("MAX ACC " + str(max(modelHistory['acc'])))
print ("AVG ACC " + str(np.mean(modelHistory['acc'])) + "\n")


########################################
