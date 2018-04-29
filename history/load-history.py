#! /usr/bin/python2.7

import sys
import sys
import pandas
import numpy as np
################################################################################################
modelHistory = np.load('./age/history_model_54.npy').item()

print ("HISTORY: ")
print (modelHistory)

print ("MAX VAL_ACC " + str(max(modelHistory['acc'])))
print ("AVG VAL_ACC " + str(np.mean(modelHistory[''])))




########################################
