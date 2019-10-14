#pragma once
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

class Filter
{
public:
	Filter();
	~Filter();
	cv::Mat inverse_gaussian_gradient(const cv::Mat& image, double alpha, double sigma, int k_size);
	cv::Mat make_init_ls(const std::pair<int, int>& img_shape, const std::pair<int, int>& circle_center, unsigned char radius);
	void gradient(const cv::Mat & img, cv::Mat& gx, cv::Mat& gy);
	cv::Mat smoothing(const cv::Mat& img);
private:
	std::vector<cv::Mat> structures;

	bool is_inf_sup_first = true;
};