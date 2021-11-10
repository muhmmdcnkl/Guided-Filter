//======================================================================
//
//�����˲�
//��˫���˲����ƣ����б�Ե�������ԣ��ڱ�ԵЧ��������˫���˲�
//�Ҹ��㷨ִ��ʱ�����˲����ڴ�С�޹أ��ʺϴ����ߴ�ͼƬ
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

	//֧�ֻҶ�����ͼ�Ͳ�ɫ����ͼ
	gf.setGuidence(img, 4, 1/255.0f);

	//�˲�ͼ������ǵ�ͨ���ģ����ǲ�ɫͼ�񣬿��Է�ͨ���˲���ϲ���������ʾ
	//�������ɫͼ���ٶȽ������ڷǵ���״̬���ܽϿ쿴�������
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
