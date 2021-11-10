#ifndef __GUIDED_FILTER_H__
#define __GUIDED_FILTER_H__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

class GuidedFilter
{
private:
	int m_radius;
	int m_rows;
	int m_cols;
	int m_border_type;
	cv::Size m_win_size;

	std::vector<cv::Mat> m_means_I;
	std::vector<cv::Mat> m_guidences;
	std::vector<cv::Mat> m_covs_I;

	std::vector<cv::Mat> m_sigmas;

private:
	void makeDepth32f(cv::Mat& source, cv::Mat& output, float scale);
	void buildVector(std::vector<cv::Mat>& element_maps, cv::Mat& out_vec, int index);
	void buildSigma(std::vector<cv::Mat>& element_maps, cv::Mat& out_sigma, int dim, int index);

	void filterSingleChannel(cv::Mat& source, cv::Mat& output, float epsilon, float scale);
	void filterMultiChannel(cv::Mat& source, cv::Mat& output, float epsilon, float scale);

public:
	void setGuidence(cv::Mat& guidence, int radius = 2, float scale = 1.0f, int border_type = cv::BORDER_DEFAULT);
	void setGuidence(std::vector<cv::Mat>& splited_guidence, int radius = 2, float scale = 1.0f, int border_type = cv::BORDER_DEFAULT);
	void filter(cv::Mat& source, cv::Mat& output, float epsilon = 0.01f, float scale = 1.0f);
};

#endif