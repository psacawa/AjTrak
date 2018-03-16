# AjTrak
Webcam Eyetracking for General Usage

To run, install most recent OpenCV, cmake, Eigen library

	$ cmake .

in the main directory, then

	$ make ImageCapture

makes a binary which allows the serial capture of training data.
Press 't' while running to start training. While training, always look at the mouse cursor and move it slowly about the screen, making sure to reach the edges and corners. 
This generates a folder `./data/training_dataXXX` for some number `XXX`. Run `TrainingModel.train (XXX, numEpochs=number)` to train on this dataset.
Finally, run `ExecuteModel.main(XXX)` to run the trained model in realtime.
