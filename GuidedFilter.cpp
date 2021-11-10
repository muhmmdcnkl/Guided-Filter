#include "GuidedFilter.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <ppl.h>

using namespace cv;
using namespace std;
using namespace concurrency;

void GuidedFilter::makeDepth32f(Mat& source, Mat& output, float scale)
{
	if (source.depth() != CV_32F || abs(scale - 1.0f) > FLT_EPSILON)
		source.convertTo(output, CV_32F, scale);
	else
		output = source;
}

void GuidedFilter::buildVector(vector<Mat>& element_maps, Mat& out_vec, int index)
{
	int dim = element_maps.size();

	float* ptr_out = (float*)out_vec.data;
	float* ptr_data;

	for (int i = 0; i < dim; ++i)
	{
		ptr_data = (float*)element_maps[i].data;
		ptr_out[i] = ptr_data[index];
	}
}

void GuidedFilter::buildSigma(vector<Mat>& element_maps, Mat& out_sigma, int dim, int index)
{
	out_sigma.create(dim, dim, CV_32FC1);

	int element_index = 0;
	float* ptr_data;
	for (int i = 0; i < dim; ++i)
	{
		for (int j = i; j < dim; ++j)
		{
			ptr_data = (float*)element_maps[element_index].data;
			out_sigma.at<float>(i, j) = ptr_data[index];
			if (i != j)
				out_sigma.at<float>(j, i) = ptr_data[index];

			++element_index;
		}
	}
}

void GuidedFilter::setGuidence(Mat& guidence, int radius /* = 2 */, float scale /* = 1.0f */, int border_type /*= BORDER_REPLICATE */)
{
	CV_Assert(radius >= 1 && scale > 0);
	CV_Assert(guidence.data != NULL);

	if (guidence.channels() > 1)
	{
		vector<Mat> splited_guidence;
		split(guidence, splited_guidence);
		setGuidence(splited_guidence, radius, scale);
		return;
	}

	m_rows = guidence.rows;
	m_cols = guidence.cols;
	m_radius = radius;
	m_border_type = border_type;
	m_win_size = Size(2*radius + 1, 2*radius + 1);

	m_guidences.resize(1);
	Mat& guidence_32f = m_guidences[0];
	m_means_I.resize(1);
	Mat& mean_I = m_means_I[0];
	m_covs_I.resize(1);
	Mat& var_I = m_covs_I[0];
	m_sigmas.clear();	//单通道不需要保存协方差矩阵


	makeDepth32f(guidence, guidence_32f, scale);
	
	//计算I*I
	Mat mat_I2;
	multiply(guidence_32f, guidence_32f, mat_I2);

	//计算I的均值和I*I的均值
	Mat mean_I2;
	boxFilter(guidence_32f, mean_I, CV_32F, m_win_size, Point(-1,-1), true, border_type);
	boxFilter(mat_I2, mean_I2, CV_32F, m_win_size, Point(-1,-1), true, border_type);

	//计算I的方差
	var_I = mean_I2 - mean_I.mul(mean_I);
}

void GuidedFilter::setGuidence( vector<Mat>& splited_guidence, int radius /* = 2 */, float scale /* = 1.0f */, int border_type /*= BORDER_REPLICATE */)
{
	CV_Assert(radius >= 1 && scale > 0);
	if (splited_guidence.size() == 1)
	{
		setGuidence(splited_guidence[0], radius, scale);
		return;
	}

	int dim = splited_guidence.size();
	m_rows = splited_guidence[0].rows;
	m_cols = splited_guidence[0].cols;
	m_radius = radius;
	m_border_type = border_type;
	m_win_size = Size(2*radius + 1, 2*radius + 1);
	int count = m_rows*m_cols;

	m_guidences.resize(dim);
	m_means_I.resize(dim);
	m_covs_I.resize(dim*(dim+1)/2);
	m_sigmas.resize(count);

	for (int i = 0; i < dim; ++i)
	{
		//保证所有参与计算的矩阵都是32位的
		makeDepth32f(splited_guidence[i], m_guidences[i], scale);
		//计算I的第i个分量图的均值
		boxFilter(m_guidences[i], m_means_I[i], CV_32F, m_win_size, Point(-1,-1), true, border_type);
	}

	//计算cov_I
	Mat mean_Ii_Ij;
	Mat mul_Ii_Ij;
	int cov_index = 0;
	for (int i = 0; i < dim; ++i)
	{
		//由于协方差矩阵是对称的，我们只需计算一半
		for (int j = i; j < dim; ++j)
		{
			mul_Ii_Ij = m_guidences[i].mul(m_guidences[j]);
			boxFilter(mul_Ii_Ij, mean_Ii_Ij, CV_32F, m_win_size, Point(-1,-1), true, border_type);
			m_covs_I[cov_index] = mean_Ii_Ij - m_means_I[i].mul(m_means_I[j]);

			++cov_index;
		}
	}

	parallel_for(0, count, [&](int i)
	{
		buildSigma(m_covs_I, m_sigmas[i], dim, i);
	});
}

void GuidedFilter::filterSingleChannel( Mat& source, Mat& output, float epsilon, float scale )
{
	Mat& guidence_32f = m_guidences[0];
	Mat& mean_I = m_means_I[0];

	Mat source_32f;
	makeDepth32f(source, source_32f, scale);

	if (guidence_32f.data == source_32f.data)
	{
		Mat source_copy;
		source_32f.copyTo(source_copy);
		source_32f = source_copy;
	}

	//计算I*p
	Mat mat_Ip;
	multiply(guidence_32f, source_32f, mat_Ip);

	//计算各种均值
	Mat mean_p, mean_Ip;
	boxFilter(source_32f, mean_p, CV_32F, m_win_size, Point(-1,-1), true, m_border_type);
	boxFilter(mat_Ip, mean_Ip, CV_32F, m_win_size, Point(-1,-1), true, m_border_type);

	//计算Ip的协方差和I的方差
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	Mat var_I = m_covs_I[0] + epsilon;

	//求a和b
	Mat a, b;
	divide(cov_Ip, var_I, a);
	b = mean_p - a.mul(mean_I);

	//对包含像素i的所有a、b做平均
	Mat mean_a, mean_b;
	boxFilter(a, mean_a, CV_32F, m_win_size, Point(-1,-1), true, m_border_type);
	boxFilter(b, mean_b, CV_32F, m_win_size, Point(-1,-1), true, m_border_type);

	//计算输出 (depth == CV_32F)
	output = mean_a.mul(guidence_32f) + mean_b;
}

void GuidedFilter::filterMultiChannel( Mat& source, Mat& output, float epsilon, float scale )
{
	int dim = m_guidences.size();
	
	vector<Mat> mat_Ips(dim);//x
	vector<Mat> mean_Ips(dim);//x
	vector<Mat> cov_Ips(dim);

	//计算p的均值
	Mat source_32f;
	Mat mean_p;
	makeDepth32f(source, source_32f, scale);
	boxFilter(source_32f, mean_p, CV_32F, m_win_size, Point(-1,-1), true, m_border_type);
	
	for (int i = 0; i < dim; ++i)
	{
		//计算I的第i个分量图与p相乘的值Ip，及其均值
		mat_Ips[i] = m_guidences[i].mul(source_32f);
		boxFilter(mat_Ips[i], mean_Ips[i], CV_32F, m_win_size, Point(-1,-1), true, m_border_type);

		//计算Ip的第i个分量图的协方差
		cov_Ips[i] = mean_Ips[i] - m_means_I[i].mul(mean_p);
	}

	//乘以epsilon后的单位阵
	Mat eU = Mat::eye(dim, dim, CV_32F)*epsilon;

	//将a的每个分量单独保存为一个图
	vector<Mat> a(dim);
	for (int i = 0; i < dim; ++i)
	{
		a[i].create(m_rows, m_cols, CV_32FC1);
	}

	int count = m_rows*m_cols;
	parallel_for(0, count, [&](int i)
	{
		Mat vec_Ip(dim, 1, CV_32FC1);
		Mat inv_sigma_eU;
		Mat a_k;

		buildVector(cov_Ips, vec_Ip, i);
		//因为是对称矩阵，所以用Cholesky分解加速
		inv_sigma_eU = (m_sigmas[i] + eU).inv(DECOMP_CHOLESKY);
		a_k = inv_sigma_eU*vec_Ip;

		//将列向量a_k的元素分别写到对应的分量图中
		for (int j = 0; j < dim; ++j)
		{
			float* ptr_a = (float*)a[j].data;
			ptr_a[i] = a_k.at<float>(j);
		}
	});

	//计算a和b的均值（以下代码顺序不能改变）
	Mat mean_b = -a[0].mul(m_means_I[0]);
	boxFilter(a[0], a[0], CV_32F, m_win_size, Point(-1,-1), true, m_border_type);
	output = a[0].mul(m_guidences[0]);
	for (int i = 1; i < dim; ++i)
	{
		mean_b -= a[i].mul(m_means_I[i]);
		//计算a的均值
		boxFilter(a[i], a[i], CV_32F, m_win_size, Point(-1,-1), true, m_border_type);
		//和对应的I分量图相乘后加到输出上
		output += a[i].mul(m_guidences[i]);
	}
	mean_b += mean_p;
	boxFilter(mean_b, mean_b, CV_32F, m_win_size, Point(-1,-1), true, m_border_type);
	output += mean_b;
}

void GuidedFilter::filter( Mat& source, Mat& output, float epsilon /*= 0.01f*/, float scale /*= 1.0f*/ )
{
	CV_Assert(source.rows == m_rows && source.cols == m_cols);
	CV_Assert(epsilon > 0 && scale > 0);
	CV_Assert(m_covs_I.size() > 0);

	if (m_covs_I.size() == 1)
	{
		filterSingleChannel(source, output, epsilon, scale);
	}
	else
	{
		filterMultiChannel(source, output, epsilon, scale);
	}
}