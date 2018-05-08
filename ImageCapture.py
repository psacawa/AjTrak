#!/usr/bin/python3

import sys, os, cv2, numpy as np
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, QGraphicsItem, 
        QDesktopWidget, QGraphicsEllipseItem, QGraphicsTextItem)
from PyQt5.QtCore import Qt , QPointF, QRectF, QRect, QThread
from PyQt5.QtGui import QPainter, QColor, QBrush, QFont

class CollectDataWidget (QGraphicsView):

    yIncr = 10
    xIncr = 20
    frameTime = 20
    outOfBounds = False
    margin = 25

    def __init__ (self):

        super (CollectDataWidget, self).__init__()
        screen = QDesktopWidget().screenGeometry ()
        self.setGeometry (screen)
        self.setWindowTitle("Data Collection")

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect (0,0, screen.right()-self.margin, screen.bottom()-self.margin)
        self.node = Node (self)
        self.scene.addItem (self.node)
        # aktualnie 3:24 z (dx,dy, lambda) = (10, 20, 20)
        text =  "Keep your eyes on the blue circle - testing once begun takes ~2 min. - press <Space> to begin"
        self.text = QGraphicsTextItem  (text)
        self.text.setPos (100, 0)
        self.scene.addItem (self.text)
        
        self.thread = CameraThread (self.node)
        self.setScene(self.scene)
        self.showFullScreen ()

    def timerEvent (self, event):
        dim = self.scene.sceneRect().bottomRight()
        node = self.node
        pos = node.scenePos()
        if pos.x() > dim.x():
            self.close()
        elif (pos.y() > dim.y() or pos.y () <= 0) and not self.outOfBounds:
            node.moveBy (abs(self.xIncr), 0)
            self.yIncr *= -1  
        
            self.outOfBounds = True
        else:
            node.moveBy (0, self.yIncr)
            if self.outOfBounds:
                self.outOfBounds = False
        

    def keyPressEvent (self, event):
        key = event.key ()
        if key == Qt.Key_Q:
            self.thread.terminate = True
            self.close()
        elif key == Qt.Key_V:
            self.node.setVisible (not self.node.isVisible())
        elif key == Qt.Key_Space:
            self.startTimer (self.frameTime)
            self.thread.start()
            self.text.setVisible (False)



class Node(QGraphicsItem):

    #  Type = QGraphicsItem.UserType + 1
    r = 10

    def __init__(self, graphWidget):
        super(Node, self).__init__()

        self.setPos (self.r, self.r)
        self.graph = graphWidget

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setZValue(1)

    def type(self):
        return Node.Type

    def shape(self):
        path = QPainterPath()
        path.addEllipse(-self.r, -self.r, 2*self.r, 2*self.r)
        return path

    def paint(self, painter, option, widget):
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.blue)
        painter.drawEllipse(-self.r, -self.r, 2*self.r, 2*self.r)

    def boundingRect(self): # ??
        adjust = 2.0
        return QRectF(-10 - adjust, -10 - adjust, 23 + adjust, 23 + adjust)

class CameraThread (QThread):

    terminate = False
    numTrainingSet = 0
    
    def __init__  (self, node):
        QThread.__init__ (self)
        self.node = node
        
    def __del__ (self):
        self.wait ()
    
    def run (self):
        targetEye = (48,32)
        dimVector = (1,)+targetEye+(3,)
        capture = cv2.VideoCapture (0)
        dataCollected = 0

        try:
            faceCascade= cv2.CascadeClassifier ("./data/haarcascade_frontalface_alt.xml")
            eyeCascade = cv2.CascadeClassifier ("./data/haarcascade_eye_tree_eyeglasses.xml")
        except Exception as e:
            print (e)
            #  self.wait() # ??
            return 

        dataFile = self.initTraining ()

        # self.terminate is set via the controlling scene's timer
        while not self.terminate:

            image = capture.read ()[1]
            imageGrayscale = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
            facesFound = faceCascade.detectMultiScale (imageGrayscale, 1.3, 5) # współczynniki ?

            if len(facesFound) == 1:
                x,y,w,h = facesFound[0]
                faceInput = np.array(facesFound[0]).reshape((1,4))
                imageFace = image[y:y+h, x:x+w]
                faceGrayscale = imageGrayscale[y:y+h, x:x+w]
                cv2.rectangle (image,(x,y), (x+w, y+h), (255,0,0), 2)
                eyesFound = eyeCascade.detectMultiScale (faceGrayscale, 1.3, 5)
                if len (eyesFound) == 2:

                    # we're in business
                    print ("Possible face detected\t", facesFound)            

                    # make sure e0 is the right eye (appears on left in the image)
                    e0, e1 = eyesFound[0], eyesFound[1]
                    if e0[0] > e1[0]:
                        e0, e1 = e1, e0
                    e0, e1 = self.scale (e0, targetEye), self.scale (e1, targetEye)

                    # experiment with local averaging
                    #  eye0Recent.push (e0)
                    #  e0 = eye0Recent.average ()
                    #  print (e0, e1)

                    # convenience rename
                    (x0,y0,w0,h0,x1,y1,w1,h1) = np.concatenate([e0,e1])
                    imageEye0 = imageFace[y0:y0+h0, x0:x0+w0]
                    imageEye1 = imageFace[y1:y1+h1, x1:x1+w1]

                    #  cv2.rectangle (imageFace,(x0,y0), (x0+w0, y0+h0), (0,0,255), 2)
                    #  cv2.rectangle (imageFace,(x1,y1), (x1+w1, y1+h1), (0,255,0), 2)

                    # write image
                    imageFilename = self.writeFolder + "/image" + str (dataCollected).zfill (4)
                    cv2.imwrite (imageFilename + ".jpg", image)
                    cv2.imwrite (imageFilename + "_eye0.jpg", imageEye0)
                    cv2.imwrite (imageFilename + "_eye1.jpg", imageEye1)

                    # write data
                    pos = self.node.scenePos()
                    dataFile.write (str (dataCollected) + ",")
                    dataFile.write (str (pos.x()) + ",")
                    dataFile.write (str (pos.y()) + ",")
                    dataFile.write (str (imageFilename) + ".jpg,")
                    dataFile.write (str (w) + "," + str (h) + ",")
                    dataFile.write (str (x) + "," + str (y) + ",")
                    dataFile.write (imageFilename + "_eye0.jpg,")
                    dataFile.write (str (w0) + "," + str (h0) + ",")
                    dataFile.write (str (x0) + "," + str (y0) + ",")
                    dataFile.write (imageFilename + "_eye1.jpg,")
                    dataFile.write (str (w1) + "," + str (h1) + ",")
                    dataFile.write (str (x1) + "," + str (y1) + "")
                    dataFile.write ("\n")
                    

                    dataCollected += 1
                else:
                    print ("{} eyes found".format (len(eyesFound)))
                    pass
            else:
                print ("{} faces found".format (len(facesFound)))
                pass

        pass

    def initTraining (self):

        # find less folder ./data/training_dataXXX which doesn't exist
        while True:
            filename = "./data/training_data{0}/data.csv".format (str(self.numTrainingSet).zfill(3))
            if os.path.isfile (filename):
                self.numTrainingSet += 1
                continue
            else:
                break

        self.writeFolder = os.path.dirname (filename)
        print ("Saving data in ", self.writeFolder)
        os.mkdir (self.writeFolder)
        dataFile = open (filename, "w")

        # write header
        variableList  = [ "data_id",
            "mouse_x" ,
            "mouse_y" ,
            "file_face" ,
            "face_height" ,
            "face_width" ,
            "face_x" ,
            "face_y" ,
            "file_eye0" ,
            "eye0_height" ,
            "eye0_width" ,
            "eye0_x" ,
            "eye0_y" ,
            "file_eye1" ,
            "eye1_height" ,
            "eye1_width" ,
            "eye1_x" ,
            "eye1_y",
        ]
        dataFile.write (",".join (variableList) + "\n")
        return dataFile

    def scale (self, r, target):
        # scale a rectangle so that (w,h) = target
        tx,ty = target
        assert ( tx % 2 == 0 and ty % 2 == 0), "Target rectangle must have even dimensions"
        centre = (r[0]+ r[2]//2, r[1] + r[3]//2)
        return np.array ((centre[0]-tx//2, centre[1]-ty//2,tx,ty))

def main ():
    app = QApplication (sys.argv)
    widget =  CollectDataWidget ()
    sys.exit (app.exec_())


if __name__ == "__main__":
    main ()
