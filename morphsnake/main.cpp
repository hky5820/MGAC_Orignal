#include <string>
#include <fstream>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "filter.h"
#include "morphsnake.h"


cv::Mat ReadMatFromTxt(std::string filename, int rows, int cols)
{
	double m;
	cv::Mat out = cv::Mat::zeros(rows, cols, CV_64FC1);//Matrix to store values

	std::ifstream fileStream(filename);
	int cnt = 0;//index starts from 0
	while (fileStream >> m)
	{
		int temprow = cnt / cols;
		int tempcol = cnt % cols;
		out.at<double>(temprow, tempcol) = m;
		cnt++;
	}
	return out;
}

// sigma 는 경계의 폭을 결정하고, 크면 넓은 범위을 gaussian 하게 되므로 경계가 모호해지고,
// 너무 작으면, 약간의 변화들도 다 경계로 나타내는 것 같다.
// alpha는 threshold 값과 서로 물려있어서, 하나를 고정해두고 나머지 하나만 컨트롤 하는게 맞는 것 같다.
// threshold가 높거나 alpha가 크게 되면, mask의 후보군이 줄어들어서 contour가 제대로 evolve 되지 못한다.
// balloon은 threshold에 물려있는데, 조금만 건드려도 threshold 값에 영향을 크게 미치므로,
// 값을 1로 두고, threshold 값을 건드리는게 더 좋아보인다.

int main() {
	cv::Mat img = cv::imread("color_.png");
	//cv::Mat img = cv::imread("body.png", cv::IMREAD_GRAYSCALE);

	Filter filter;
	MorphSnake ms;

	cv::Mat bgr[3]; cv::split(img, bgr);

	double alpha = 4000;
	double sigma = 3;
	int bgr_or_gray = 0;
	int iteration = 200;
	int smoothing = 5;
	double threshold = 0.3;
	int ballon = 1;
	double downscale = 4;


	cv::Mat gray;
	if (bgr_or_gray < 3) {
		gray = bgr[bgr_or_gray];
	}
	else {
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
	}
	
	gray.convertTo(img, CV_64FC1, 1./255.);

	//cv::Mat origin_gimg = ReadMatFromTxt("gimg.txt", img.rows, img.cols);
	cv::Mat	gimg = filter.inverse_gaussian_gradient(img, alpha, sigma, 25);
	
	
	cv::Mat init_ls = filter.make_init_ls({gimg.rows/ downscale, gimg.cols/ downscale},
		{img.rows / ( downscale * 2), img.cols / ( downscale * 2 ) }, 
		img.rows  / (10 * downscale) );
	
	cv::Mat mask = ms.morphological_geodesic_active_contour(gimg, iteration, 
		init_ls, smoothing, threshold, ballon, downscale);
	
	cv::imshow("mask", mask);
	cv::waitKey(0);

	return 0;
}
