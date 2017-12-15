#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <stdio.h>

using std::cout;
using std::endl;
using std::string;
using namespace cv;

int main (int argc, char* argv[]) {
	string filename, outputFilename ;
	char io[100];
	int numImagesSaved = 0;
	const cv::String keys =
		"{h||no help for you}"
		"{@opt|<none>|opcja}"
		"{file|<none>|}";
	cv::CommandLineParser parser (argc, argv, keys);
	cv::Mat image, imageGray, gradX, gradY;
	if (parser.has ("h")) {
		parser.printMessage();
		image = cv::imread(filename, cv::IMREAD_COLOR);
	}
	if (parser.has("file"))
		filename = parser.get <string> ("file");
	else {
		cv::VideoCapture capture;
		capture.open( 0,cv::CAP_V4L2   );
		if (!capture.isOpened() ) {
			printf("--(!)Error opening video capture\n");
			return -1;
		}
		capture.read (image);
	}


	for (;;numImagesSaved++) {
		sprintf (io, "out/out%.3d.jpg", numImagesSaved);
		cout << io << endl;
		if (access (io, F_OK) == -1)
			break;
	}
	outputFilename = string (io);

	cv::cvtColor (image,imageGray, cv::COLOR_BGR2GRAY);

    cv::Mat kernel = (cv::Mat_<int>(3,3) <<  0, 1,  0, 1, -4, 1, 0, 1,  0);
	cv::Mat kernelGradX = (cv::Mat_<int>(1,3) << 1,0,1);
	cv::Mat kernelGradY = (cv::Mat_<int>(3,1) << 1,0,1);
	cv::Rect_<int> rect(0,0,4,4);
//	cout << gradMagnitude.rows << ' ' << gradMagnitude.cols
//		 << ' ' << gradMagnitude.channels() << endl;;
	cv::filter2D (imageGray, gradX, image.depth(), kernelGradX);
	cv::filter2D (imageGray, gradY, image.depth(), kernelGradY);
	cout << imageGray (rect) << endl << gradX(rect) << endl;

	cv::Mat gradMagnitude = (cv::Mat_<float> (imageGray.size()));
	for (int i = 0; i != gradMagnitude.rows ; i++) {
		for (int j= 0; j != gradMagnitude.cols; j++)
			gradMagnitude.at<float>(i,j) = std::sqrt (
				std::pow (gradX.at<char> (i,j), 2) + std::pow (gradY.at<char>(i,j),2));
	}
	imwrite (outputFilename, gradX);
	imshow ("output" ,gradMagnitude);
	cv::waitKey();
}
