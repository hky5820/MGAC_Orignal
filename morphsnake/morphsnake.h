#pragma once
#include <opencv2/core.hpp>

#include "filter.h"

class MorphSnake
{
public:
	MorphSnake();
	~MorphSnake();
	cv::Mat morphological_geodesic_active_contour(const cv::Mat& gimg_input,
		int iterations,
		const cv::Mat& init_ls,
		int smoothing,
		double threshold,
		int ballon,
		double downscale);

private:
	Filter *filter;
};

