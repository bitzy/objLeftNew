#include "objLeftDetect.h"
#include <opencv2/opencv.hpp> //cvCapture
#include <opencv2/core/core.hpp> //getTickFrequency
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;
#define DEBUG 

int main()
{
	char test_video[200] = "test.avi";
	//read video info
	CvCapture *videoCap;
	videoCap = cvCaptureFromFile(test_video);
	//videoCap = cvCaptureFromCAM(0);
	double _width = cvGetCaptureProperty(videoCap, CV_CAP_PROP_FRAME_WIDTH);
	double _heigh = cvGetCaptureProperty(videoCap, CV_CAP_PROP_FRAME_HEIGHT);
	int _fps = (int)cvGetCaptureProperty(videoCap, CV_CAP_PROP_FPS);
	int _frameNum = (int)cvGetCaptureProperty(videoCap, CV_CAP_PROP_FRAME_COUNT);
	cvSetCaptureProperty(videoCap, CV_CAP_PROP_POS_FRAMES, 0);//at the begin.

	//choose focus area
	CvPoint pointArray1[4];
	CvPoint *pointArray[2] = { &pointArray1[0], &pointArray1[2] };		
	pointArray[0][0] = cvPoint(20, 16);
	pointArray[0][1] = cvPoint(592, 52);
	pointArray[1][0] = cvPoint(564, 448);
	pointArray[1][1] = cvPoint(46, 440);
	IplImage * mask = cvCreateImage(cvSize(_width, _heigh), IPL_DEPTH_8U, 1);
	int arr[1] = { 4 };
	cvPolyLine(mask, pointArray, arr, 1, 5, CV_RGB(255, 255, 255));
	cvFillPoly(mask, pointArray, arr, 1, CV_RGB(255, 255, 255));
	cvThreshold(mask, mask, 254, 255, CV_THRESH_BINARY);
	
	//read video & process
	ObjLeftDetect _objleft(mask);
	bool obj_left = false;
	IplImage *qImg;
	int count = 0;
	Rect pos;
	while (qImg = cvQueryFrame(videoCap)) {
		Mat _qImg(qImg);
		int64 work_begin = getTickCount();
		obj_left = _objleft.process(qImg, pos);		
		if (obj_left == true) {
			/*printf("%d,%d,%d,%d\n", pos.x, pos.y, pos.width, pos.height);
			rectangle(_qImg, pos, Scalar(255, 0, 0), 2);
			imwrite("prob.jpg", _qImg);*/
			printf("alarm!!!\n");
		}
		int64 delta = getTickCount() - work_begin;
		double work_fps = getTickFrequency() / delta;
		stringstream ss;
		ss << work_fps;
		putText(_qImg, "fps:" + ss.str(), Point(5, 20), FONT_HERSHEY_SIMPLEX, 
			0.9, Scalar(255, 100, 0), 2);
		cvShowImage("video", qImg);
		cvWaitKey(1);
		count++;
	}
	cvReleaseImage(&qImg);
	cvReleaseImage(&mask);
	return 0;
}