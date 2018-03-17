# na razie podwaliny tylko działają
# błędne wychwytywanie pozycji oka

import xdo
import numpy as np
import cv2
from Xlib import display
import matplotlib.pyplot as plt
# forces the code to run on CPU
# must be executed before tf/keras imported
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
from keras import models
from Common import *

# trzeba będzie chyba zaimplementować przednie przejście w tensorflow
# nie wygląda na to, że keras ma takie możliwości
import tensorflow as tf

# run the specified model online
def main (modelId='last',moveMouse=False):
    targetEye = (48,32)
    dimVector = (1,)+targetEye+(3,)
    recurrent = True

    model = getTrainedModel (modelId)
    model.summary()

    if recurrent:
        config = model.get_config()
        config = configForwardPass (config)
        model.from_config (config)
        model.summary()


    xContext = xdo.Xdo()
    capture = cv2.VideoCapture (0)
    faceCascade= cv2.CascadeClassifier ("./haarcascade_frontalface_alt.xml")
    eyeCascade = cv2.CascadeClassifier ("./haarcascade_eye_tree_eyeglasses.xml")
    eye0Recent = LastAverage (10)

    try:
        cv2.namedWindow ("test")
        #  cv2.namedWindow ("test2")
        while True:
            ch = cv2.waitKey(1) & 0xFF
            if ch == ord ('q'):
                break
            image = capture.read ()[1]
           
            imageGrayscale = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
            facesFound = faceCascade.detectMultiScale (imageGrayscale, 1.3, 5) # współczynniki ?

            if len(facesFound) == 1:
                #  print (facesFound)
                x,y,w,h = facesFound[0]
                faceInput = np.array(facesFound[0]).reshape((1,4))
                imageFace = image[y:y+h, x:x+w]
                faceGrayscale = imageGrayscale[y:y+h, x:x+w]
                cv2.rectangle (image,(x,y), (x+w, y+h), (255,0,0), 2)
                eyesFound = eyeCascade.detectMultiScale (faceGrayscale, 1.3, 5)
                if len (eyesFound) == 2:
                    #  print ("Possible face detected", facesFound)            
                    e0, e1 = eyesFound[0], eyesFound[1]
                    if e0[0] > e1[0]:
                        e0, e1 = e1, e0
                    e0, e1 = scale (e0, targetEye), scale (e1, targetEye)

                    # experiment with local averaging
                    eye0Recent.push (e0)
                    e0 = eye0Recent.average ()

                    (x0,y0,w0,h0,x1,y1,w1,h1) = np.concatenate([e0,e1])
                    cv2.rectangle (imageFace,(x0,y0), (x0+w0, y0+h0), (0,0,255), 2)
                    cv2.rectangle (imageFace,(x1,y1), (x1+w1, y1+h1), (0,255,0), 2)
                    imageEye0 = imageFace[y0:y0+h0, x0:x0+w0]
                    imageEye1 = imageFace[y1:y1+h1, x1:x1+w1]
                    eyeInput = np.concatenate (
                       (imageEye0.reshape (dimVector).transpose(0, 2,1,3),
                       imageEye1.reshape (dimVector).transpose(0, 2,1,3)),
                       axis=3
                    )

                    # forward pass
                    modelPrediction = model.predict (x=[faceInput,eyeInput])
                    modelPrediction = modelPrediction.flatten()
                    mouseLoc = getMousePos ()
                    predictionError = np.linalg.norm (modelPrediction-mouseLoc)
                    print ("Delta: {}\tPredicted: {}\tActual: {}".format(
                        "%.2f" % predictionError,modelPrediction,mouseLoc))
                    if moveMouse:
                        xContext.move_mouse (modelPrediction[0],modelPrediction[1],0)
                else:
                    #  print ("{} eyes found".format (len(eyesFound)))
                    pass

            else:
                print ("{} faces found".format (len(facesFound)))
                pass

            cv2.imshow ("test", image )
            #  cv2.imshow ("test2", imageEye0)
    except Exception as e:
        print (e)
        pass

    capture.release()
    cv2.destroyAllWindows()

def configForwardPass (config):
    for i, layer in enumerate (config['layers']):
        if layer['class_name'] == 'InputLayer':
            input_shape = layer['config']['batch_input_shape']
            #  print (input_shape)
            layer['batch_input_shape'] = tuple ([1]+list(input_shape))
            #  print (layer['batch_input_shape'] )
        if 'stateful' in  layer['config']:
            layer['config']['stateful'] = True
    return config

def scale (r, target):
    # scale a rectangle so that (w,h) = target
    tx,ty = target
    assert ( tx % 2 == 0 and ty % 2 == 0), "Target rectangle must have even dimensions"
    centre = (r[0]+ r[2]//2, r[1] + r[3]//2)
    return np.array ((centre[0]-tx//2, centre[1]-ty//2,tx,ty))

def getMousePos():
    # mousepos() --> (x, y) get the mouse coordinates on the screen (linux, Xlib)
    data = display.Display().screen().root.query_pointer()._data
    return np.array((data["root_x"], data["root_y"]))

class LastAverage ():
    # class for recalling the last N values of an arithmetic type
    # and returning the average of them in O(1) time

    def __init__ (self, N):
        self.length = N
        self.accepted = 0
        self.cumulative = 0
        self.data = [0] * N

    def average (self):
        return (self.cumulative / min (self.length, self.accepted)).astype(int)

    def push (self,x):
        self.cumulative -= self.data[self.accepted % self.length]
        self.data[self.accepted % self.length] = x
        self.cumulative += self.data[self.accepted % self.length]
        self.accepted+=1

if __name__ == "__main__":
    main ()
