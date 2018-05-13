// opencv
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
// systemowe wezwania
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
// prosta grafika
#include <cairo/cairo.h>
#include "Timer.h"
// ten extern jest potrzebny skoro xdo to biblioteka C
// masz lepszy pomysł to daj znać
extern "C" {
#include <xdo.h>
}

#include <iostream>
#include <cstdio>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;
using std::swap;


enum detectMode {off, search, lock, relock, debug, train};

int initDisplay ();
void initTraining ();
void detectAndDisplay ();
void displayCascade (cv::Mat frame, const vector <cv::Rect> &faces,
									const vector <vector<cv::Rect>> &eyes);
void mprintf (string str);
void outputRect (std::ostream &ostr, cv::Rect &r, char del = ' ', char newl = '\n');
void processData ();
cv::Rect adjoinRect (const cv::Rect &delta, const :: cv::Rect &r);
cv::Rect scale (const cv::Rect r, int w, int h);
void printAggregate ();

string faceCascadeName, eyesCascadeName;
const string windowName = "Camera";
cv::CascadeClassifier faceCascade;
cv::CascadeClassifier eyesCascade;
cv::VideoCapture capture;

detectMode currentMode = search;
Timer modeTimer;
const double lockInterval = 1.5;
bool verbose = false;

cv::Mat frame;
cv::Rect lockFace;
vector <cv::Rect> lockEyes;

vector <cv::Rect> facesFound;
vector <vector <cv::Rect>> eyesFound;

xdo_t *xContext;
int mouseX, mouseY, mouseScr;
ofstream file;
int dataCollected = 0, numTrainingSets;
string outputFolder, outputFilename;

// zmienne tymczasowe
double sumEyeWidth, sumEyeHeight, sumEyeXDiff, sumEyeYDiff;
double sumFaceWidth, sumFaceHeight;
const int testingDiscrminationThreshold = 10;
int recordEyeWidth = 48, recordEyeHeight = 32;

int main (int argc, char *argv[])
{
	cv::CommandLineParser parser = cv::CommandLineParser (argc, argv,
		"{help h||}"
		"{faceCascade|haarcascade_frontalface_alt.xml|}"
		"{eyesCascade|haarcascade_eye_tree_eyeglasses.xml|}"
		"{eye32|false|capture eyes with size 32*32}"
	);
	faceCascadeName = parser.get<string>("faceCascade");
	eyesCascadeName = parser.get<string>("eyesCascade");
	cout << "Face cascade from " << faceCascadeName << endl
		 << "Eye cascade from  " << eyesCascadeName << endl;
	if (parser.get<bool> ("eye32"))
		recordEyeWidth = recordEyeHeight = 32;

	if 	(initDisplay ()  == -1) {
		cout << "Problem initializing display" << endl;
		return 1;
	}

	// int f = 0;
	for (;capture.read (frame);)
	{
		if (frame.empty()) {
			printf ("Frame capture failed.\n");
			break;
		}

		char ch = (char) cv::waitKey (1);
		switch (ch) {
			// case 'f':
			//     cout << "hallo " << f++ << endl;
			//     break;
			case 'q':
			case 27:
				goto esc_loop;
				break;
			case 'd':
				if (currentMode == detectMode::debug)
					currentMode = detectMode::search;
				else
					currentMode = detectMode::debug;
				break;
			case 'v':
				verbose = !verbose;
				break;
			case 't':
				// cv::destroyAllWindows();
				if (currentMode == detectMode::train) {
					file.close ();
					currentMode = detectMode::debug;
					cv::namedWindow (windowName);
					printf ("Data written to data/training_data%.3d\n", numTrainingSets);
				} else {
					initTraining ();
					currentMode = detectMode::train;
					// cv::namedWindow ("Train", cv::WINDOW_FULLSCREEN);
				}
			default:
				break;
		}

		detectAndDisplay ();
	}
	esc_loop:;
	xdo_free (xContext);
	printAggregate ();
}

void detectAndDisplay ()
{
	cv::Mat frameGrayscale;
	cv::cvtColor (frame, frameGrayscale, cv::COLOR_BGR2GRAY);

	switch (currentMode) {
		case detectMode::debug:
		case detectMode::search:
		case detectMode::train: {
			mprintf("Search mode\n");
			faceCascade.detectMultiScale (frameGrayscale, facesFound);
			for (unsigned i = 0; i != facesFound.size (); i++) {
				cv::Rect &face = facesFound[i];
				eyesFound.push_back({});
				eyesCascade.detectMultiScale (frameGrayscale (face), eyesFound[i]);
			}
			break;
		}
		case detectMode::lock:
			mprintf ("Lock mode\n");
			eyesCascade.detectMultiScale (frameGrayscale (lockFace), lockEyes);
			break;
		default:
			break;
	}

	switch (currentMode) {
		case detectMode::debug:
		case detectMode::search:
		case detectMode::train: // zlikwidować póżdniej...
			displayCascade (frame, facesFound, eyesFound);
			break;
		case detectMode::lock:
			displayCascade (frame, {lockFace}, {lockEyes});
			break;
			break;
		default:
			break;
	}

	switch (currentMode) {
		case detectMode::search:
			if (facesFound.size () == 1 && eyesFound[0].size () == 2) {
				modeTimer.start (lockInterval);
				lockFace = facesFound[0];
				lockEyes = eyesFound[0];
				currentMode = detectMode::lock;
			}
			break;
		case detectMode::lock:
			if (modeTimer.up()) {
				currentMode = detectMode::search;
			}
			break;
		case detectMode::train:
			xdo_get_mouse_location (xContext, &mouseX, &mouseY, &mouseScr);
			// cout << mouseX << ' ' << mouseY << endl;
			if (facesFound.size () == 1 && eyesFound[0].size () == 2) {
				processData();
				dataCollected++;
			}
			break;
		default:
			break;
	}
}


int initDisplay ()
{
	//-- 1. Load the cascades
	if (!faceCascade.load( "data/" + faceCascadeName ) ){
		printf("--(!)Error loading face cascade\n");
		return -1;
	};
	if (!eyesCascade.load( "data/" +  eyesCascadeName ) ){
		printf("--(!)Error loading eyes cascade\n");
		return -1;
	};
	//-- 2. Read the video stream
	capture.open( 0,cv::CAP_V4L2   );
	if (!capture.isOpened() ) {
		printf("--(!)Error opening video capture\n");
		return -1;
	}

	xContext = xdo_new (nullptr);
	cv::namedWindow (windowName, cv::WINDOW_AUTOSIZE);

	return 0;
}

void initTraining () {
	struct stat sb;
	numTrainingSets = 0;
	char folder [40];
	for (;;numTrainingSets++) {
		sprintf (folder,"data/training_data%.3d", numTrainingSets);
		if (stat(folder, &sb) != 0 || !S_ISDIR(sb.st_mode))
			break;
	}
	mkdir (folder, 0700);
	cout << folder << endl;;
	outputFolder = string(folder) + string (1,'/');
	outputFilename = outputFolder + "data.csv";
	file.open (outputFilename);
	dataCollected = 0;
	// inicjalizacja zmiennych
	sumEyeWidth = sumEyeHeight = sumEyeXDiff = sumEyeYDiff = 0;
	sumFaceWidth = sumFaceHeight = 0;
	file << "data_id" << ','
		 << "mouse_x" << ','
		 << "mouse_y" << ','
	  	 << "file_face" << ','
		 << "face_height" << ','
		 << "face_width" << ','
		 << "face_x" << ','
		 << "face_y" << ','
		 << "file_eye0" << ','
		 << "eye0_height" << ','
		 << "eye0_width" << ','
		 << "eye0_x" << ','
		 << "eye0_y" << ','
		 << "file_eye1" << ','
		 << "eye1_height" << ','
		 << "eye1_width" << ','
		 << "eye1_x" << ','
		 << "eye1_y" << '\n';
	return;
}

void displayCascade (cv::Mat frame, const vector <cv::Rect> &faces,
									const vector <vector<cv::Rect>> &eyes) {
	cv::Mat canvas = frame.clone();
	for (unsigned i = 0; i != faces.size(); i++) {
		cv::rectangle (canvas, faces[i], cv::Scalar (255,0,0), 4,8,0);
		for (const cv::Rect &eye : eyes[i]) {
			cv::Rect loc = cv::Rect (
				faces[i].x+eye.x, faces[i].y+eye.y, eye.width, eye.height);
			// trzeba tu naprawić położenie prostokąta dla oczu
			cv::rectangle (canvas, loc, cv::Scalar (0,255,0), 4,8,0);
		}
	}
	cv::imshow (windowName, canvas);
}

void mprintf (string str) {
	if (verbose)
		printf ("%s", str.data());
}

void outputRect (std::ostream &ostr, cv::Rect &r, char del , char newl ) {
	ostr << r.height << del << r.width << del << r.x << del << r.y << newl;
}

cv::Rect getRect (ifstream &istr) {
	int h, w, x, y;
	istr >> h >> w >> x >> y;
	return cv::Rect (x,y, w, h);
}

void processData () {
	cv::Rect &f = facesFound[0];
	cv::Rect &e0 = eyesFound [0][0], &e1 = eyesFound [0][1];
	if (e0.x > e1.x)
		swap (e0, e1);

	cout << "Found possible face\n";
	outputRect (cout ,f);
	outputRect (cout ,e0);
	outputRect (cout ,e1);

	// uaktualnienie danych
	sumFaceWidth +=  f.width;
	sumFaceHeight +=  f.height;
	sumEyeWidth += e0.width + e1.width;
	sumEyeHeight += e0.height + e1.height;
	sumEyeXDiff += e1.x - e0.x;
	sumEyeYDiff += abs (e1.y - e0.y);

	char cstr [30];
	sprintf (cstr, "image%.4d", dataCollected);
	string imageFilename = cstr;
	string pathToImage = outputFolder + imageFilename+ ".jpg";
	// cout << pathToImage << endl;

	// zapis danych
	file << dataCollected << ',';
	file << mouseX << ',' << mouseY << ',';

	file << imageFilename + ".jpg" << ',';
	outputRect (file, facesFound[0], ',', ',');
	cv::imwrite (pathToImage, frame);
	for (int c = 0; c!= 2; c++) {
		string eyeImageFilename = imageFilename+ "_eye" + string (1,'0'+c);
		string pathToEyeImage = outputFolder+eyeImageFilename;
		file << eyeImageFilename + ".jpg" << ',';
		outputRect (file, eyesFound[0][c], ',', c ? '\n' : ',');
		// file << imageFilename << ',';
		cv::imwrite (pathToEyeImage+ ".jpg",
			frame (scale (adjoinRect (facesFound[0], eyesFound[0][c]), recordEyeWidth, recordEyeHeight)));
	}
	// file << '\n';
}

cv::Rect adjoinRect (const cv::Rect &delta, const :: cv::Rect &r) {
	return cv::Rect (delta.x+ r.x, delta.y+r.y, r.width, r.height);
}

cv::Rect scale (const cv::Rect r, int w, int h) {
	assert (w % 2 == 0 && h % 2 == 0);
	int cx = r.x + r.width/2, cy = r.y + r.height/2;
	return cv::Rect (cx - w/2, cy - h/2, w, h);
}


void printAggregate () {
	cout << "Aggregate data\n";
	cout << "Number of faces collected: " << dataCollected << endl;
	cout << "Avg. face width/height\n";
	cout << sumFaceWidth / dataCollected << ' ' << sumFaceHeight / dataCollected << endl;
	cout << "Avg. eye width/height\n";
	cout << sumEyeWidth / (2*dataCollected) << ' ' << sumEyeHeight / (2*dataCollected) << endl;
	cout << "Avg. eye difference x/y\n";
	cout << sumEyeXDiff / (2*dataCollected) << ' ' << sumEyeYDiff / (2*dataCollected) << endl;
}


