def getModelFilename (modelId):
    folder = "./models/"
    if type(modelId) == list or type(modelId) == range:
        modelFile = str(list(modelId))
    else:
        modelFile = str(modelId).zfill(3)
    modelFile = folder + "model" + modelFile + ".hdf5"
    return modelFile
