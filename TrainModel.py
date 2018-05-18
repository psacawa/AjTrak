import pandas as pd
import keras
from keras import layers, optimizers, losses, models, regularizers, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout, LSTM
from keras.models import Sequential 
from keras.utils import plot_model
from skimage import io,transform
from Common import getModelFilename
import numpy as np
from time import time
#  import tensorflow as tf


def train (datasetId,modelId=2,numEpoch=100,initModel=None):
    # władować model
    if initModel != None:
        model = initModel
    else:
        model = retrieveModel (modelId)
    print (model.input_shape)
    model.summary()
    plot_model (model,  "./images/model.png")

    # władować dane
    faceData, eyeData, mouseData = loadData(datasetId, model.input_shape[-1])
    print ("Face Data  : ", faceData.shape)
    print ("Eye Data  : ",  eyeData.shape)
    print ("Labels: ",      mouseData.shape)

    # ustawienia optymalizacji
    optimizer = optimizers.Adam(lr=1e-3)
    loss = losses.mean_squared_error

    # trenować model
    model.compile(optimizer=optimizer, loss=loss)
    startTime = time()
    model.fit (x=[faceData,eyeData], y=mouseData, epochs=numEpoch, batch_size=8, shuffle=True)
    print ("Training took {} seconds".format (time() - startTime))

    # zapisać model
    modelFolder = "./models/" 
    models.save_model (model,  getModelFilename (datasetId))
    return model

def trainRecurrent (datasetId,modelId=3,numEpoch=100):
    # trenowąć modele rekurencyjne z oddzielaniem danych w batche
    model = retrieveModel (modelId)
    model.summary()
    plot_model (model,  "./images/model.png")

    faceData, eyeData, mouseData = loadData(datasetId, model.input_shape[-1])
    faceData  = batchSplit (faceData,step    = batchSize)
    eyeData   = batchSplit ( eyeData, step   = batchSize)
    mouseData = batchSplit ( mouseData, step = batchSize)
    print ("Face Data  : ", faceData.shape)
    print ("Eye Data  : ",  eyeData.shape)
    print ("Labels: ",      mouseData.shape)

# obtain models
def retrieveModel (i= 2):
    if i == 0:
        # wyłącznie korzystając z eye0, czyli z prawego oka
        return Sequential([
            Conv2D (96, 3, activation='relu', data_format="channels_last", input_shape=(32,48,3)),
            MaxPooling2D(),
            Conv2D (256, 3, activation='relu', data_format="channels_last"),
            MaxPooling2D(),
            Conv2D (512, 3, activation='relu', data_format="channels_last"),
            Flatten(),
            Dense (16, activation='relu'),
            Dense(2)
        ]) 
    elif i == 1:
        # bez współrzędnych odpowiadające twarzy
        return  Sequential([
            Conv2D (96, 6, activation='relu', data_format="channels_last", input_shape=(32,48,6)),
            MaxPooling2D(),
            Conv2D (256, 3, activation='relu', data_format="channels_last"),
            MaxPooling2D(),
            Conv2D (512, 3, activation='relu', data_format="channels_last"),
            Flatten(),
            Dense (16, activation='relu'),
            Dense(2)
        ])
    elif i == 2:
        eye  = Input (shape=(32,48,6,))
        face = Input (shape = (4,))

        eyeTensor = eye
        eyeTensor = Conv2D (96, 6, activation='relu', data_format="channels_last") (eyeTensor)
        eyeTensor = MaxPooling2D () (eyeTensor)
        eyeTensor = Conv2D (256, 3, activation='relu', data_format="channels_last") (eyeTensor)
        eyeTensor = Dropout (rate=9.3) (eyeTensor)
        eyeTensor = MaxPooling2D () (eyeTensor)
        eyeTensor = Conv2D (512, 3, activation='relu', data_format="channels_last") (eyeTensor)
        eyeTensor = Flatten ()(eyeTensor)

        faceTensor = face
        faceTensor = Dense (32, activation = 'relu') (faceTensor)

        merged = concatenate (inputs=[faceTensor,eyeTensor], axis=1)
        #  merged = Dense (16, activation='relu') (merged)
        merged = Dense (16, activation='relu') (merged)
        #  merged = Dense (16) (merged)
        merged = Dense (2) (merged)
        return Model (inputs=[face,eye] , outputs= merged)
    elif i == 3:
        eyeInp  = Input (shape=(None,32,48,6,))
        faceInp = Input (shape = (None,4))
        eye = Flatten () (eyeInp)
       
        #  merged = concatenate ([eyeInp
    elif i == 4:
        # recurrent
        eyeInp  = Input (shape=(None,32,48,6,))
        faceInp = Input (shape = (4,))

        eye  = eyeInp
        eye  = Conv2D (96, 6, activation  = 'relu', data_format = "channels_last") (eye)
        eye  = MaxPooling2D () (eye)
        eye  = Conv2D (256, 3, activation = 'relu', data_format = "channels_last") (eye)
        eye  = Dropout (rate              = 9.3) (eye)
        eye  = MaxPooling2D () (eye)
        eye  = Conv2D (512, 3, activation = 'relu', data_format = "channels_last") (eye)
        eye  = Flatten ()(eye)
        eye  = LSTM (128)(eye)
        face = faceInp

        merged = concatenate (inputs=[face,eye], axis=1)
        merged = Dense (128, activation='relu')
        merged = LSTM (32) (merged)
        merged = Dense (2)
        return Model (inputs=[faceInp, eyeInp], outputs =merged)



def loadDataSet (i,shape=(1,32,48,6)):
# ładować pojedynczy zbiór danych
    folder = "data/" + "training_data" + str(i).zfill (3) + "/"
    frame = pd.read_csv(folder + "data.csv")
    mouse_x = frame.mouse_x.as_matrix()
    mouse_y = frame.mouse_y.as_matrix()
    mouse = np.asmatrix ([mouse_x,mouse_y]).transpose()
    #  eye0 = np.array ([io.imread (folder + string).transpose() for string in frame.file_eye0])
    #  eye1 = np.array ([io.imread (folder + string).transpose() for string in frame.file_eye1])
    eye0 = np.array ([io.imread (folder + string) for string in frame.file_eye0])
    eye1 = np.array ([io.imread (folder + string) for string in frame.file_eye1])
    face = np.array (frame.loc[:,['face_x', 'face_y', 'face_width','face_height']])

    # jeśli model używa oboje oczy, to zwróć skonkatenowane oczy wzdłuż wymiaru kanałów ( [-1] )
    if shape[-1] == 6:
        eye = np.concatenate ((eye0, eye1), axis=3)
        return face,eye, mouse
    else:
        return face,eye0, mouse

def loadData (l, shape=(1,32,48,6), batchSize=None):
# ladować albo pojedynczy zbiór danych, reprezentowane przez int, albo i lista
# możliwe użycie opcji batchSize, które rozdziela kazde ze zbiorów w rozdzielone batche o tej długości
    shape = list(shape)
    shape[0] = 0
    if type (l) == list or type (l) == range:
        faceData = np.empty (shape=(0,4))
        eyeData = np.empty (shape)
        mouseData = np.empty (shape=(0,2))
        for i in l:
            print ("Loading dataset " + str (i).zfill (3))
            newFace, newEye, newLabels = loadDataSet (i,shape)
            faceData =   np.concatenate ([faceData, newFace],axis=0)
            eyeData =    np.concatenate ([eyeData, newEye],axis=0) 
            mouseData =    np.concatenate ((mouseData, newLabels),axis= 0)
        return faceData, eyeData, mouseData
    elif type (l) == int:
        return loadDataSet (l, shape)

# funkcja odzielającą dane w batchy o określonej długości, 
# która to będzie długością trenowanych batchów
def batchSplit (data, batchSize=100):
    ret = np.split (data, np.arange(data.shape[0],step=batchSize)[1:])[:-1]
    #  print ([x.shape for x in ret])
    return np.array (ret)

def main (modelFile=""):
    if modelFile == "":
        print ("Enter file to save to")
        input (modelFile)
    print ("Enter set to train on")
    input (n)
    print ("Enter number of epochs")
    input (epochs)
    model = train (n, numEpoch=epochs)

if __name__ == "__main__":
    main()
