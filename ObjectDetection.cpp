#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <eigen3/Eigen/Dense>
#include "Timer.h"

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <unistd.h>
#include <limits>
#include <queue>

using std::vector;
using std::cout;
using std::endl;
using std::string;
using std::queue;
using Eigen::Vector4d;
using std::numeric_limits;

using namespace std::chrono;
using namespace cv;

enum detectMode {off, search, lock, relock, debug};

/** Function Headers */
void detectAndDisplay( Mat frame );
void detect (Mat frame);
int initDisplay ();
void printRect (const Rect& r);
Point findPupil (Mat eye);

/** Global variables */
String faceCascadeName, eyesCascadeName;
CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;
String windowName = "Capture - Face detection";
const double lockInterval = 1.5;
const double permissibleRelockError = 125.0;
const unsigned framesToRelock = 5;
const unsigned framesToSearch = 10;
detectMode mode = detectMode::search;
VideoCapture capture;
double timeSpentReading = 0.,timeSpentCascading = 0.;
bool drawCascade = true;
bool pupilFlag = false;
bool printFrames = true;
Timer modeTimer, frameTimer;

int frameCountThisSecond= 0;
int lockFrames = 0;
int lockFrameTwoEyes = 0;
unsigned framesInMode;

Rect lockFace;
Mat lockFaceROI;
vector <Rect> lockEyes;

double blurSigmaX, blurSigmaY;
bool blurFrame = false;

/** searchMode */


/** @function main */
int main( int argc, const char** argv )
{
	char filename [50];
	unsigned frameCount = 0, numScreenshotsTaken = 0;
	CommandLineParser parser = CommandLineParser (argc, argv,
		"{help h||}"
		"{faceCascade|haarcascade_frontalface_alt.xml|}"
		"{eyesCascade|haarcascade_eye_tree_eyeglasses.xml|}"
		"{sigma|0.0|}"
		"{blur|0|}");

	if (argc > 1 && !strcmp (argv[1],   "help")) {
		cout << "\nThis program demonstrates using \
		the cv::CascadeClassifier class to detect \
		objects (Face + eyes) in a video stream.\n"
				"You can use Haar or LBP features.\n\n";
		parser.printMessage();
	}
	faceCascadeName = parser.get<string>("faceCascade");
	eyesCascadeName = parser.get<string>("eyesCascade");
	if (parser.get<int> ("sigma") != 0.0) {
		blurSigmaX = blurSigmaY =parser.get<int> ("sigma");
		blurFrame = true;
	}
	cout.precision (3);
	cout << "Face cascade from " << faceCascadeName << endl
		 << "Eye cascade from  " << eyesCascadeName << endl;

	if 	(initDisplay ()  == -1) {
		cout << "Problem initializing display" << endl;
		return 1;
	}

	// znajdź liczbę już wykonanych zrzutów ekranu
	for (;;numScreenshotsTaken++) {
		sprintf (filename, "filename%.3d.jpg", numScreenshotsTaken);
		if (access (filename, F_OK) == -1)
			break;
	}

	frameTimer.start (1.0);
	Mat frame;

	for (;capture.read(frame); frameCount++) {

		if(frame.empty() ) {
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		//-- 3. Apply the classifier to the frame
		//detectAndDisplay( frame );
		detectAndDisplay (frame);

		// wprowadzanie komend poprzez klawiaturę
		char c = (char)waitKey(1);
		switch (c) {
			case 'q':
			case 27:
				goto esc_loop;
			case 'd':
				mode = detectMode::debug;
				break;
			case 's':
				mode = detectMode::search;
				break;
			case 't':
				printFrames = !printFrames;
			case 'g':
				pupilFlag = true;
			case 'p':
				cout << "Face:\n";
				printRect (lockFace);
				cout << "Eyes: " << lockEyes.size () << endl;
				for (auto e: lockEyes)
					printRect (e);
				break;
			case ' ':
				sprintf (filename, "filename%.3d.jpg", numScreenshotsTaken);
				imwrite (filename, frame);
				cout << "Saved to " << filename << endl;
				for (unsigned i = 0; i != lockEyes.size () ; ++i) {
					sprintf (filename, "eye%.3d-%d.jpg", numScreenshotsTaken, i);
					imwrite (filename, frame(lockFace)(lockEyes[i]));
				}
				cout << "Saved eyes" << endl;
				numScreenshotsTaken++;
				break;
		}

		if (frameTimer.up ()) {
			if (printFrames)
				cout << "This second " << frameCountThisSecond
					 << " frames" << endl;
			frameCountThisSecond = 0;
			timeSpentCascading = 0.0;
			frameTimer.reset();
		}
		frameCountThisSecond++;
	}
	esc_loop:
	cout << lockFrames << " klatek, z których " << lockFrameTwoEyes <<
		" z dwoma oczami" << endl;
	// only for personal use
//	system ("wmctrl -a demiurg");
	return 0;
}

int initDisplay ()
{
	//-- 1. Load the cascades
	if (!faceCascade.load( faceCascadeName ) ){
		printf("--(!)Error loading face cascade\n");
		return -1;
	};
	if (!eyesCascade.load( eyesCascadeName ) ){
		printf("--(!)Error loading eyes cascade\n");
		return -1;
	};
	//-- 2. Read the video stream
	capture.open( 0,cv::CAP_V4L2   );
	if (!capture.isOpened() ) {
		printf("--(!)Error opening video capture\n");
		return -1;
	}
	return 0;
}

void detectAndDisplay (Mat frame)
{
	Mat frameGrayscale;
	Mat frameGrayBlur;
	static Mat &frameToPrint = (blurFrame) ? frameGrayBlur : frame;
	cvtColor( frame, frameGrayscale, COLOR_BGR2GRAY );
	if (blurFrame)
		GaussianBlur (frameGrayscale, frameGrayBlur, Size (0,0), blurSigmaX, blurSigmaY);
	equalizeHist( frameGrayscale, frameGrayscale );

	vector <Rect> newFaces;
	vector <Mat> newFacesROI;
	vector <vector <Rect>> newEyes;

	// Pozyskanie obrazów i kaskadowanie
	switch (mode) {
		case detectMode::debug:
		case detectMode::relock:
		case detectMode::search:
//			cout << "Detecting faces" << endl;
			faceCascade.detectMultiScale (frameGrayscale, newFaces,
				1.1,2., 0|CASCADE_SCALE_IMAGE, Size (30,30));
			for (auto r : newFaces) {
				newEyes.push_back ({});
				newFacesROI.push_back (frameGrayscale (r));
				eyesCascade.detectMultiScale( newFacesROI.back(),
					newEyes.back (), 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
			}
			if (drawCascade){
//				cout << "Malowanie" << endl;
				for (unsigned i = 0 ; i != newFaces.size (); i++) {
					Rect &f = newFaces [i];
					rectangle (frame, f, Scalar (0,255,255), 4,8,0);
					for (auto e : newEyes[i]) {
						Point tl (f.tl () + e.tl ());
						Rect eye (tl, e.size());
						rectangle (frame, eye, Scalar (0, 255,0), 4, 8, 0);
					}
				}
			}
			break;
		case detectMode::lock:
			eyesCascade.detectMultiScale(frameGrayscale(lockFace),
				lockEyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
			lockFrames++;
			if (lockEyes.size () == 2) lockFrameTwoEyes++;
			if (drawCascade) {
				rectangle (frame, lockFace, Scalar (0,255,255), 4,8,0);
//				cout << "Malowanie w lock, " <<  lockEyes.size () << " oczy\n";
				for (auto e : lockEyes) {
//					cout << e.width << ' ';
					Point tl (lockFace.tl () + e.tl ());
					Rect eye (tl, e.size());
					// findPupil (frameGrayscale (eye));
					rectangle (frame, eye, Scalar (0, 255,0), 4, 8, 0);
				}
//				cout << endl;
			}
			break;
		default:
			break;
	}

	framesInMode++;
	// przetwarzanie specifyczne dla trybów
	switch (mode) {
		case detectMode::lock:
//			cout << "Obecnie minęło " << modeTimer.current ()<< endl;
			if (modeTimer.up ()) {
				mode = detectMode::relock;
				framesInMode = 0;
				cout << "Wejście do trybu \t\trelock" << endl;
			}
			if (pupilFlag) {
//				for (auto e : lockEyes) {
//					;
//				}
				pupilFlag = false;
			}
			break;
		case detectMode::search:
//			cout << "W trybie search: " << newFaces.size () << " twarzy\n";
			if (newFaces.size () == 1) {
				lockFace = newFaces [0];
				mode = detectMode::lock;
				// zainicjować odliczanie czasu w lock
				modeTimer.start (lockInterval);
				framesInMode = 0;
				cout << "Wejście do trybu \t\tlock" << endl;
			}
			break;
		case detectMode::relock: {
			double lowestError = numeric_limits<double>::infinity();
			Vector4d lockFaceVec (lockFace.x, lockFace.y,
				lockFace.width, lockFace.height);
			unsigned candidateFaceIndex;
			for (unsigned c = 0; c != newFaces.size (); c++) {
				auto f = newFaces [c];
				Vector4d candidateVec  (f.x, f.y, f.width, f.height);
				if ((lockFaceVec-candidateVec).norm () < lowestError) {
					lowestError = (lockFaceVec-candidateVec).norm ();
					candidateFaceIndex = c;
				}
			}
			cout << "Najmniesza różnica to " << lowestError << endl;
			if (lowestError < permissibleRelockError &&
					newEyes[candidateFaceIndex].size () >= 2) {
				lockFace = newFaces[candidateFaceIndex];
				mode = detectMode::lock;
				framesInMode = 0;
				modeTimer.start (lockInterval);
				cout << "Wejście do trybu \t\tlock" << endl;
			}
			if (framesInMode >= framesToRelock) {
				mode = detectMode::search;
				framesInMode = 0;
				cout << "Wejście do trybu \t\tsearch" << endl;
			}
			break;
		} default:
			break;
	}

	imshow( windowName, frameToPrint);
}

Point findPupil (Mat eye) {
	Mat kernelGradX = (Mat_<int>(3,1) << -1,0,1),
		kernelGradY = (Mat_<int>(1,3) << -1,0,1), grad, gradBlurred;
	cout << kernelGradX.rows << ' ' << kernelGradX.cols << ' ' <<
		kernelGradX.channels () << endl;
	cout <<eye.rows << ' ' << eye.cols << ' ' << eye.channels () << endl;
	cv::filter2D (eye, grad, eye.depth(), kernelGradX);
	GaussianBlur (grad, gradBlurred , Size (0,0), blurSigmaX, blurSigmaY);
	cv::imshow ("blurred", gradBlurred);
	cv::waitKey ();
	return Point(0,0);
}

void printRect (const Rect& r) {
	cout 	<< "x=" << r.x << " y=" << r.y <<
			 " ht=" << r.height << " wd=" << r.width << endl;
}


/** @function detectAndDisplay */
//void detectAndDisplay(Mat frame)
//{
//	static std::vector<Rect> faces;
//	std::vector<Rect> newFaces;
//	static auto lastFaceRead = system_clock::now ();
//	Mat frameGrayscale;
//
//	cvtColor( frame, frameGrayscale, COLOR_BGR2GRAY );
//	equalizeHist( frameGrayscale, frameGrayscale );
//
//	auto before = system_clock::now ();
//
//	//-- Detect faces
//	duration <double> diff = system_clock::now() - lastFaceRead;
//	if ((system_clock::now() - lastFaceRead).count () > faceInterval) {
//		cout << "Odkrywanie twarzy" << endl;
//		faceCascade.detectMultiScale( frameGrayscale, newFaces,
//				1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
//		if (mode == detectMode::lock && faces.size () > 0) {
//			Rect old_face = faces[0];
//			double min_dist = numeric_limits<double>::infinity();
//			Rect& closestFace = newFaces[0];
//			for (auto f : newFaces) {
//				auto center = [](Rect& r)
//						{return Point (r.x + r.width/2., r.y+ r.height/2.);};
//				double dist = norm (center (old_face) - center (f));
//				if (dist < min_dist) {
//					closestFace = f;
//					min_dist = dist;
//				}
//			}
//			faces.clear ();
//			faces.push_back (closestFace);
//		} else if (mode == detectMode::lock) {
//			faces.clear ();
//			faces.push_back (newFaces[0]);
//		} else
//			faces = newFaces;
//		lastFaceRead = system_clock::now();
//	}
//
//	for ( size_t i = 0; i < faces.size(); i++ ) {
//		rectangle (frame, faces[i], Scalar (0,255,255), 4,8,0);
//
//		//-- In each face, detect eyes
//		Mat faceROI = frameGrayscale( faces[i] );
//		std::vector<Rect> eyes;
//		eyesCascade.detectMultiScale( faceROI,
//			eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
//
//		for ( size_t j = 0; j < eyes.size(); j++ ) {
//			Point tl (faces[i].tl () + eyes[j].tl ());
//			Rect eye (tl, eyes[j].size());
//			rectangle (frame, eye, Scalar (0, 255,0), 4, 8, 0);
//		}
//
//	}
//	diff = system_clock::now () - before;
//	timeSpentCascading += diff.count();
//	//-- Show what you got
//	imshow( windowName, frame );
//}
