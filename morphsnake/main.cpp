#include <string>
#include <fstream>
#include <iostream>

#include <opencv2/highgui.hpp>

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

// sigma �� ����� ���� �����ϰ�, ũ�� ���� ������ gaussian �ϰ� �ǹǷ� ��谡 ��ȣ������,
// �ʹ� ������, �ణ�� ��ȭ�鵵 �� ���� ��Ÿ���� �� ����.
// alpha�� threshold ���� ���� �����־, �ϳ��� �����صΰ� ������ �ϳ��� ��Ʈ�� �ϴ°� �´� �� ����.
// threshold�� ���ų� alpha�� ũ�� �Ǹ�, mask�� �ĺ����� �پ�� contour�� ����� evolve ���� ���Ѵ�.
// balloon�� threshold�� �����ִµ�, ���ݸ� �ǵ���� threshold ���� ������ ũ�� ��ġ�Ƿ�,
// ���� 1�� �ΰ�, threshold ���� �ǵ帮�°� �� ���ƺ��δ�.

int main() {

	Filter filter;
	MorphSnake ms;
	cv::Mat img_in = cv::imread("body.png");
	cv::Mat bgr[3]; cv::split(img_in, bgr);
	cv::Mat img = bgr[2];
	img.convertTo(img, CV_64FC1, 1./255.);

	//cv::Mat origin_gimg = ReadMatFromTxt("gimg.txt", img.rows, img.cols);
	cv::Mat	gimg = filter.inverse_gaussian_gradient(img, 4000, 3, 25);
	
	
	
	double downscale = 2;
	cv::Mat init_ls = filter.make_init_ls({gimg.rows/ downscale, gimg.cols/ downscale},
		{img.rows / ( downscale * 2), img.cols / ( downscale * 2 ) }, 
		img.rows  / (10 * downscale) );
	
	cv::Mat mask = ms.morphological_geodesic_active_contour(gimg, 300, init_ls, 5, 0.3, 1, downscale);
	
	cv::imshow("mask", mask);
	cv::waitKey(0);

	return 0;
}
