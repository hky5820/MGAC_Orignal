#include <string>   // for type2str >> remove
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "filter.h"

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

Filter::Filter() {
	structures.resize(4);
	for (int i = 0; i < 4; i++) structures[i] = cv::Mat::zeros(3, 3, CV_8UC1);
	for (int k = 0; k < 3; k++) {
		structures[0].at<uchar>(k, k) = 1;
		structures[2].at<uchar>(k, 2 - k) = 1;
		structures[1].at<uchar>(1, k) = 1;
		structures[3].at<uchar>(k, 1) = 1;
	}
}

Filter::~Filter() {
}

cv::Mat Filter::inverse_gaussian_gradient(const cv::Mat & image, double alpha, double sigma, int k_size){
	cv::Mat img = image;
	cv::Mat blur;
	int rows = img.rows, cols = img.cols;

	cv::Mat gx, gy;
	cv::GaussianBlur(img, blur, cv::Size(k_size, k_size), sigma);

	gradient(blur, gx, gy);

	cv::Mat output = cv::Mat::zeros(rows, cols, blur.type());

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double val = sqrtl(powl(gx.at<double>(i, j), 2.) + powl(gy.at<double>(i, j), 2.));
			output.at<double>(i, j) = 1. / sqrtl(1. + alpha * val);
		}
	}

	return output;
}

cv::Mat Filter::make_init_ls(const std::pair<int, int>& img_shape, const std::pair<int, int>& circle_center, unsigned char radius){

	int rows = img_shape.first;
	int cols = img_shape.second;

	cv::Mat output = cv::Mat::zeros(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double val = radius - sqrtl(powl(i - circle_center.first, 2.) + powl(j - circle_center.second, 2.));
			if (val > 0)
				output.at<uchar>(i, j) = 255;
		}
	}

	return output;
}

void Filter::gradient(const cv::Mat & img, cv::Mat& gx, cv::Mat& gy){
	gx.release(); gy.release();

	int rows = img.rows, cols = img.cols;
	gx = cv::Mat::zeros(rows, cols, CV_64FC1);
	gy = cv::Mat::zeros(rows, cols, CV_64FC1);

	for (int r = 1; r < rows - 1; r++) {
		for (int c = 1; c < cols - 1; c++) {
			gx.at<double>(r, c) = (img.at<double>(r, c + 1) - img.at<double>(r, c - 1)) / 2.;
			gy.at<double>(r, c) = (img.at<double>(r + 1, c) - img.at<double>(r - 1, c)) / 2.;
		}
	}
}

cv::Mat Filter::smoothing(const cv::Mat & img){
	int rows = img.rows, cols = img.cols;
	std::vector<cv::Mat> temps(4);
	for (int i = 0; i < 4; i++) {
		temps[i] = cv::Mat::zeros(rows, cols, CV_8UC1);
	}

	cv::Mat t = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::Mat output = cv::Mat::zeros(rows, cols, CV_8UC1);
	if (is_inf_sup_first) {
		for (int i = 0; i < 4; i++) cv::dilate(img, temps[i], structures[i]);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if ( (temps[0].at<uchar>(r, c) != 0)
					&& (temps[1].at<uchar>(r, c) != 0)
					&& (temps[2].at<uchar>(r, c) != 0)
					&& (temps[3].at<uchar>(r, c) != 0)) {
					t.at<uchar>(r, c) = 255;
				}
			}
		}
		for (int i = 0; i < 4; i++) cv::erode(t, temps[i], structures[i]);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if ((temps[0].at<uchar>(r, c) != 0)
					|| (temps[1].at<uchar>(r, c) != 0)
					|| (temps[2].at<uchar>(r, c) != 0)
					|| (temps[3].at<uchar>(r, c) != 0)) {
					output.at<uchar>(r, c) = 255;
				}
			}
		}
	}
	else {
		for (int i = 0; i < 4; i++) cv::erode(img, temps[i], structures[i]);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if ((temps[0].at<uchar>(r, c) != 0)
					|| (temps[1].at<uchar>(r, c) != 0)
					|| (temps[2].at<uchar>(r, c) != 0)
					|| (temps[3].at<uchar>(r, c) != 0)) {
					t.at<uchar>(r, c) = 255;
				}
			}
		}
		for (int i = 0; i < 4; i++) cv::dilate(t, temps[i], structures[i]);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if ((temps[0].at<uchar>(r, c) != 0)
					&& (temps[1].at<uchar>(r, c) != 0)
					&& (temps[2].at<uchar>(r, c) != 0)
					&& (temps[3].at<uchar>(r, c) != 0)) {
					output.at<uchar>(r, c) = 255;
				}
			}
		}
	}
	is_inf_sup_first = !is_inf_sup_first;

	return output;
}