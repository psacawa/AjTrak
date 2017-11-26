#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <Eigen/Dense>
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

/** Global variables */
String faceCascadeName, eyesCascadeName;
CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;
String windowName = "Capture - Face detection";
const double lockInterval = 1.5;
const double permissibleRelockError = 100.0;
detectMode mode = detectMode::search;
VideoCapture capture;
double timeSpentReading = 0.,timeSpentCascading = 0.;
bool drawCascade = true;
Timer modeTimer, frameTimer;

int frameCountThisSecond= 0;
int lockFrames = 0;
int lockFrameTwoEyes = 0;


/** searchMode */


/** @function main */
int main( int argc, const char** argv )
{
	char filename [50];
	unsigned frameCount = 0, numScreenshotsTaken = 0;
	CommandLineParser parser = CommandLineParser (argc, argv,
		"{help h||}"
		"{faceCascade|haarcascade_frontalface_alt.xml|}"
		"{eyesCascade|haarcascade_eye_tree_eyeglasses.xml|}");

	if (argc > 1 && !strcmp (argv[1],   "help")) {
		cout << "\nThis program demonstrates using \
		the cv::CascadeClassifier class to detect \
		objects (Face + eyes) in a video stream.\n"
				"You can use Haar or LBP features.\n\n";
		parser.printMessage();
	}
	faceCascadeName = parser.get<string>("faceCascade");
	eyesCascadeName = parser.get<string>("eyesCascade");
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

	frameTimer.start (1.);

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
			case ' ':
				sprintf (filename, "filename%.3d.jpg", numScreenshotsTaken++);
				imwrite (filename, frame);
				cout << "Saved to " << filename << endl;
				break;
		}

		if (frameTimer.up ()) {
			cout << "This second " << frameCountThisSecond << " frames, ";
			frameCountThisSecond = 0;
			timeSpentCascading = 0.0;
		}

	}
	esc_loop:
	cout << lockFrames << " klatek, z których " << lockFrameTwoEyes <<
		" z dwoma oczami" << endl;
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
	capture.open( 0 );
	if (!capture.isOpened() ) {
		printf("--(!)Error opening video capture\n");
		return -1;
	}
	return 0;
}

void detectAndDisplay (Mat frame)
{
	Mat frameGrayscale;
	cvtColor( frame, frameGrayscale, COLOR_BGR2GRAY );
	equalizeHist( frameGrayscale, frameGrayscale );

	vector <Rect> newFaces;
	vector <Mat> newFacesROI;
	vector <vector <Rect>> newEyes;

	static Rect lockFace;
	Mat lockFaceROI;
	vector <Rect> lockEyes;

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
				cout << "Malowanie w lock, " <<  lockEyes.size () << " oczy\n";
				for (auto e : lockEyes) {
					Point tl (lockFace.tl () + e.tl ());
					Rect eye (tl, e.size());
					rectangle (frame, eye, Scalar (0, 255,0), 4, 8, 0);
				}
			}
			break;
		default:
			break;
	}

	// przetwarzanie specifyczne dla trybów
	switch (mode) {
		case detectMode::lock:
//			cout << "Obecnie minęło " << modeTimer.current ()<< endl;
			if (modeTimer.up ()) {
				mode = detectMode::relock;
				cout << "Wejście do trybu relock" << endl;
			}
			break;
		case detectMode::search:
			cout << "W trybie search: " << newFaces.size () << " twarzy\n";
			if (newFaces.size () == 1) {
				lockFace = newFaces [0];
				mode = detectMode::lock;
				// zainicjować odliczanie czasu w lock
				modeTimer.start (lockInterval);
				cout << "Wejście do trybu lock" << endl;
			}
			break;
		case detectMode::relock: {
			double lowestError = numeric_limits<double>::infinity();
			Vector4d lockFaceVec (lockFace.x, lockFace.y,
				lockFace.width, lockFace.height);
			Rect candidateFace;
			for (auto f : newFaces ) {
				Vector4d candidateVec  (f.x, f.y, f.width, f.height);
				if ((lockFaceVec-candidateVec).norm () < lowestError) {
					lowestError = (lockFaceVec-candidateVec).norm ();
					candidateFace = f;
				}
			}
			cout << "Najmniesza różnica to " << lowestError << endl;
			if (lowestError < permissibleRelockError) {
				lockFace = candidateFace;
				mode = detectMode::lock;
				cout << "Wejście do trybu lock" << endl;
				modeTimer.start (lockInterval);
			}
			break;
		} default:
			break;
	}


	imshow( windowName, frame );
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
