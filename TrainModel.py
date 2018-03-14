import pandas as pd
import keras
from keras import layers, optimizers, losses, models
from keras.models import Sequential 
import numpy as np
from skimage import io,transform
from Common import *
#  import tensorflow as tf


def train (datasetId,modelId=0,numEpoch=100):
    model = getModel (modelId)
    print (model.input_shape)
    model.summary()

    data, labels = loadData(datasetId,model.input_shape)
    print ("Data  : ", data.shape)
    print ("Labels: ", labels.shape)

    optimizer = optimizers.Adam(lr=1e-3)
    loss = losses.mean_squared_error
    model.compile(optimizer=optimizer, loss=loss)
    model.fit (data, labels, epochs=numEpoch, batch_size=8)
    modelFolder = "./models/" 
    models.save_model (model,  getModelFilename (datasetId))
    return model

def getModel (i= 0):
    return {
        # wyłącznie korzystając z eye0, czyli z prawego oka
        0: Sequential([
            layers.Conv2D (96, 3, activation='relu', data_format="channels_last", input_shape=(32,48,3)),
            layers.MaxPool2D(),
            layers.Conv2D (256, 3, activation='relu', data_format="channels_last"),
            layers.MaxPool2D(),
            layers.Conv2D (512, 3, activation='relu', data_format="channels_last"),
            #  layers.Conv2D (512, 3, activation='relu', data_format="channels_last"),
            layers.Flatten(),
            layers.Dense (16, activation='relu'),
            layers.Dense(2)
        ]), 
        # to samo co getModel (0), tylko, zo użyciem obu oczy
        1: Sequential([
            layers.Conv2D (96, 6, activation='relu', data_format="channels_last", input_shape=(32,48,6)),
            layers.MaxPool2D(),
            layers.Conv2D (256, 3, activation='relu', data_format="channels_last"),
            layers.MaxPool2D(),
            layers.Conv2D (512, 3, activation='relu', data_format="channels_last"),
            #  layers.Conv2D (512, 3, activation='relu', data_format="channels_last"),
            layers.Flatten(),
            layers.Dense (16, activation='relu'),
            layers.Dense(2)
        ]), 
        2: Sequential([
            layers.Conv2D (96, 3, activation='relu', data_format="channels_last", input_shape=(32,96,3)),
            layers.MaxPool2D()
        ])
    }.get(i, 0)

def loadDataSet (i,shape=(1,32,48,3)):
    folder = "data/" + "training_data" + str(i).zfill (3) + "/"
    frame = pd.read_csv(folder + "data.csv")
    mouse_x = frame.mouse_x.as_matrix()
    mouse_y = frame.mouse_y.as_matrix()
    mouse = np.asmatrix ([mouse_x,mouse_y]).transpose()
    #  eye0 = np.array ([io.imread (folder + string).transpose() for string in frame.file_eye0])
    #  eye1 = np.array ([io.imread (folder + string).transpose() for string in frame.file_eye1])
    eye0 = np.array ([io.imread (folder + string) for string in frame.file_eye0])
    eye1 = np.array ([io.imread (folder + string) for string in frame.file_eye1])

    # jeśli model używa oboje oczy, to zwróć skonkatenowane oczy wzdłuż wymiaru kanałów ( [-1] )
    if shape[-1] == 6:
        eye = np.concatenate ((eye0, eye1), axis=3)
        return eye, mouse
    else:
        return eye0, mouse

def loadData (l, shape=(1,32,48,3)):
    shape = list (shape)
    shape[0] = 0
    if type (l) == list or type (l) == range:
        data = np.empty (shape)
        labels= np.empty (shape=(0,2))
        for i in l:
            print ("Loading dataset " + str (i).zfill (3))
            newData, newLabels = loadDataSet (i,shape)
            data = np.concatenate ((data, newData),axis=0)
            labels = np.concatenate ((labels, newLabels),axis= 0)
        return data, labels
    elif type (l) == int:
        return loadDataSet (l, shape)

def main (modelFile=""):
    if modelFile == "":
        print ("Enter file to save to")
        input (modelFile)
    print ("Enter set to train on")
    input (n)
    print ("Enter number of epochs")
    input (epochs)
    model = train (n, numEpoch=epochs)
