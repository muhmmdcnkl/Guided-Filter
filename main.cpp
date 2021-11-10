//======================================================================
//
//引导滤波
//与双边滤波类似，具有边缘保持特性，在边缘效果上优于双边滤波
//且该算法执行时间与滤波窗口大小无关，适合处理大尺寸图片
//
//======================================================================

#include "GuidedFilter.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	TickMeter tm;
	GuidedFilter gf;

	Mat img = imread("cat.jpg");
	//Mat img = imread("girl.jpg");

	//支持灰度引导图和彩色引导图
	gf.setGuidence(img, 4, 1/255.0f);

	//滤波图像必须是单通道的，若是彩色图像，可以分通道滤波后合并，如下所示
	//（处理彩色图像速度较慢，在非调试状态下能较快看到结果）
	vector<Mat> img_channel;
	split(img, img_channel);
	vector<Mat> res(3);

	tm.start();
	gf.filter(img_channel[0], res[0], 0.05f, 1/255.0f);
	gf.filter(img_channel[1], res[1], 0.05f, 1/255.0f);
	gf.filter(img_channel[2], res[2], 0.05f, 1/255.0f);
	tm.stop();

	cout<<tm.getTimeMilli()<<endl;

	Mat output;
	merge(res, output);
	output.convertTo(output, CV_8UC3, 255);

	imshow("Result", output);

	waitKey();

	return 0;
}
