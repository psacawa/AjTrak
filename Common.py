import os

def getModelFilename (modelId='last'):
    folder = "./models/"
    if modelId == 'last':
        fileModTimes = [os.path.getmtime (folder + file) for file in os.listdir (folder)]
        indMostRecent = max (enumerate (fileModTimes), key=lambda p : p[1])[0]
        modelFile = os.listdir (folder)[indMostRecent]
        print (indMostRecent, modelFile)
    elif type(modelId) == list or type(modelId) == range:
        modelFile = "model" + str(list(modelId)).replace(' ','')+ ".hdf5"
    else:
        modelFile = "model" + str(modelId).zfill(3)+ ".hdf5"
    modelFile = folder + modelFile 
    return modelFile

def getTrainedModel (modelId='last'):
    try:
        modelFile = getModelFilename (modelId)
        model = models.load_model (modelFile) 
        return model
    except:
        print ("Model file " + modelFile + " not found")
        return
