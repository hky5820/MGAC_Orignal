#include <opencv2/imgproc.hpp> // morphological operation
#include <opencv2/highgui.hpp> // imshow, waitkey ...

#include "morphsnake.h"

MorphSnake::MorphSnake() : filter(new Filter()){
}

MorphSnake::~MorphSnake(){
}

cv::Mat MorphSnake::morphological_geodesic_active_contour(const cv::Mat & gimg_input, 
														  int iterations, const cv::Mat & init_ls, 
														  int smoothing, double threshold, int ballon,
														  double downscale	){
	cv::Mat gimg;
	cv::resize(gimg_input, gimg, cv::Size(gimg_input.cols / downscale, gimg_input.rows / downscale), 0, 0, CV_INTER_NN);
	int rows = gimg.rows, cols = gimg.cols;
	
	cv::Mat gx, gy;
	filter->gradient(gimg, gx, gy);

	cv::Mat threshold_mask_balloon = cv::Mat::zeros(rows, cols, CV_8UC1);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (gimg.at<double>(i, j) > (threshold / fabsl(ballon))) {
				threshold_mask_balloon.at<uchar>(i, j) = 255;
			}
		}
	}

	cv::Mat u = init_ls;
	cv::Mat structure = cv::Mat::ones(3, 3, CV_8UC1);

	for (int c_itr = 0; c_itr < iterations; c_itr++) {
		cv::Mat aux;
		if (ballon > 0) {
			cv::dilate(u, aux, structure);
		}
		else if (ballon < 0) {
			cv::erode(u, aux, structure);
		}
		if (ballon != 0) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (threshold_mask_balloon.at<uchar>(i, j)) {
						u.at<uchar>(i, j) = aux.at<uchar>(i, j);
					}
				}
			}
		}

		cv::Mat dgx, dgy;
		cv::Mat du;
		u.convertTo(du, CV_64FC1, 1./255.);

		filter->gradient(du, dgx, dgy);
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				double val = gx.at<double>(i, j) * dgx.at<double>(i, j) 
					+ gy.at<double>(i, j) * dgy.at<double>(i, j);
				if (val > 0) {
					u.at<uchar>(i, j) = 255;
				}
				else if (val < 0) {
					u.at<uchar>(i, j) = 0;
				}
			}
		}
		for (int i = 0; i < smoothing; i++) {
			u = filter->smoothing(u);
		}

		cv::imshow("u",u);
		cv::waitKey(1);
	}
	cv::resize(u, u, cv::Size(u.cols * downscale, u.rows * downscale), 0, 0, CV_INTER_NN);
	return u;
}
