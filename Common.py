import os
import json
import numpy as np
from keras import models
from Xlib import display

def getMousePos():
    # mousepos() --> (x, y) get the mouse coordinates on the screen (linux, Xlib)
    data = display.Display().screen().root.query_pointer()._data
    return np.array((data["root_x"], data["root_y"]))

def getModelFilename (modelId='last'):
    folder = "./models/"
    if modelId == 'last':
        fileModTimes = [os.path.getmtime (folder + file) for file in os.listdir (folder)]
        indMostRecent = max (enumerate (fileModTimes), key=lambda p : p[1])[0]
        modelFile = os.listdir (folder)[indMostRecent]
        #  print (indMostRecent, modelFile)
    elif type(modelId) == list or type(modelId) == range:
        modelFile = "model" + str(list(modelId)).replace(' ','')+ ".hdf5"
    else:
        modelFile = "model" + str(modelId).zfill(3)+ ".hdf5"
    modelFile = folder + modelFile 
    return modelFile

def getTrainedModel (modelId='last'):
    try:
        modelFile = getModelFilename (modelId)
        print ("Loading model file {}".format (modelFile))
        model = models.load_model (modelFile) 
        return model
    except Exception as e:
        print (str(e))
        print ("Model file " + modelFile + " not found")
        return

def printLayer (layer):
    print (json.dumps (layer['config'],sort_keys= True, indent=4))

def printConfig (conf):
    print (json.dumps (conf,sort_keys= True, indent=4))
