#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <unistd.h>
#include <climits>

using std::vector;
using std::cout;
using std::endl;
using std::string;
using namespace std::chrono;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
int initDisplay ();
enum detectMode {off, search, lock, relock, debug};

/** Global variables */
String faceCascadeName, eyesCascadeName;
CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;
String windowName = "Capture - Face detection";
const double faceInterval = 1.5;
detectMode mode = detectMode::debug;
VideoCapture capture;
double timeSpentReading = 0.,timeSpentCascading = 0.;

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

	Mat frame;
	time_point <system_clock> lastSecond = system_clock::now(), currentSecond;
	system_clock::time_point timeBeforeRead = system_clock::now (), timeAfterRead;

	for (;capture.read(frame); frameCount++) {
		timeAfterRead = system_clock::now ();
		duration <double>diff = timeAfterRead - timeBeforeRead;
		timeSpentReading += diff.count ();

		if(frame.empty() ) {
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		//-- 3. Apply the classifier to the frame
		detectAndDisplay( frame );
		imshow (windowName, frame);

		// wprowadzanie komend poprzez klawiaturę
		char c = (char)waitKey(10);
		switch (c) {
			case 'q':
			case 27:
				goto esc_loop;
			case 'd':
				mode = detectMode::debug;
				break;
			case 'n':
				mode = detectMode::search;
				break;
			case ' ':
				sprintf (filename, "filename%.3d.jpg", numScreenshotsTaken++);
				imwrite (filename, frame);
				cout << "Saved to " << filename << endl;
				break;
		}

		if ((system_clock::now () -lastSecond).count () > 1.) {
			lastSecond = currentSecond;
			cout << "This second " << frameCount << " frames, " << timeSpentReading<<
				"s used for read and " << timeSpentCascading <<
				"s used for cascade detection" << endl;
			frameCount = 0;
			timeSpentCascading = 0.0;
		}
		timeBeforeRead = system_clock::now ();
	}
	esc_loop:
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

void detect (Mat frame)
{
	Mat frameGrayscale;
	cvtColor( frame, frameGrayscale, COLOR_BGR2GRAY );
	equalizeHist( frameGrayscale, frameGrayscale );

	statics vector <Rect>;
	switch (mode) {
		case detectMode::debug:
		case detectMode::search:
		case detectMode::lock:
	}
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	static std::vector<Rect> faces;
	std::vector<Rect> newFaces;
	static auto lastFaceRead = system_clock::now ();
	Mat frameGrayscale;

	cvtColor( frame, frameGrayscale, COLOR_BGR2GRAY );
	equalizeHist( frameGrayscale, frameGrayscale );

	auto before = system_clock::now ();

	//-- Detect faces
	duration <double> diff = system_clock::now() - lastFaceRead;
	if ((system_clock::now() - lastFaceRead).count () > faceInterval) {
		cout << "Odkrywanie twarzy" << endl;
		faceCascade.detectMultiScale( frameGrayscale, newFaces,
				1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
		if (mode == detectMode::lock && faces.size () > 0) {
			Rect old_face = faces[0];
			double min_dist = DBL_MAX;
			Rect& closestFace = newFaces[0];
			for (auto f : newFaces) {
				auto center = [](Rect& r)
						{return Point (r.x + r.width/2., r.y+ r.height/2.);};
				double dist = norm (center (old_face) - center (f));
				if (dist < min_dist) {
					closestFace = f;
					min_dist = dist;
				}
			}
			faces.clear ();
			faces.push_back (closestFace);
		} else if (mode == detectMode::lock) {
			faces.clear ();
			faces.push_back (newFaces[0]);
		} else
			faces = newFaces;
		lastFaceRead = system_clock::now();
	}

	for ( size_t i = 0; i < faces.size(); i++ ) {
		rectangle (frame, faces[i], Scalar (0,255,255), 4,8,0);

		//-- In each face, detect eyes
		Mat faceROI = frameGrayscale( faces[i] );
		std::vector<Rect> eyes;
		eyesCascade.detectMultiScale( faceROI,
			eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

		for ( size_t j = 0; j < eyes.size(); j++ ) {
			Point tl (faces[i].tl () + eyes[j].tl ());
			Rect eye (tl, eyes[j].size());
			rectangle (frame, eye, Scalar (0, 255,0), 4, 8, 0);
		}

	}
	diff = system_clock::now () - before;
	timeSpentCascading += diff.count();
	//-- Show what you got
	imshow( windowName, frame );
}

