#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;

struct myColor {
	unsigned char B; unsigned char G;  unsigned char R;
};

template <class T, T V>
struct constant
{
	operator T() const { return V; }
};
class ConnectedComponents
{
	struct Similarity {
		int id, sameas, tag;
		Similarity() : id(0), sameas(0) {}
		Similarity(int _id, int _sameas) : id(_id), sameas(_sameas) {}
		Similarity(int _id) : id(_id), sameas(_id) {}
	};
	std::vector<Similarity> labels;
	int highest_label;
public:
	ConnectedComponents(int soft_maxlabels) : labels(soft_maxlabels) {
		clear();
	}
	void clear() {
		std::fill(labels.begin(), labels.end(), Similarity());
		highest_label = 0;
	}
	template<class Tin, class Tlabel, class Comparator, class Boolean>
	int connected(const Tin *img, Tlabel *out,
		int width, int height, Comparator,Boolean K8_connectivity);
private:
	bool is_root_label(int id) {
		return (labels[id].sameas == id);
	}
	int root_of(int id) {
		while (!is_root_label(id)) {
			labels[id].sameas = labels[labels[id].sameas].sameas;
			id = labels[id].sameas;
		}
		return id;
	}
	bool is_equivalent(int id, int as) {
		return (root_of(id) == root_of(as));
	}
	bool merge(int id1, int id2) {
		if (!is_equivalent(id1, id2)) {
			labels[root_of(id1)].sameas = root_of(id2);
			return false;
		}
		return true;
	}
	int new_label() {
		if (highest_label + 1 > labels.size())
			labels.reserve(highest_label * 2);
		labels.resize(highest_label + 1);
		labels[highest_label] = Similarity(highest_label);
		return highest_label++;
	}	
	template<class Tlabel>
	int relabel_image(Tlabel *out, int width, int height);	
};

class myImage {		
public:
	int myw, myh, myd;
	unsigned char * myData;
	myImage(int w, int h, int d):myw(w), myh(h), myd(d) {
		//Memory_Allocate1D:
		myData = (unsigned char *)malloc(myw*myh*myd*sizeof(unsigned char));
		for (int i = 0; i < myw*myh*myd; i++) myData[i] = 0;
	}
	void myReleaseImage(){ free(myData); }	

	void myImg_zero();
	void myImg_inverse(myImage* output);
	void myImg_and(myImage * input2, myImage * output);
	void myImg_Dilation(int n);
	void myImg_copy(myImage * output);
	void myImg_rgb2gray(myImage * output);
	void myImg_resize(myImage* output);
	void formMyImg(IplImage * input);	
	void trans2IPL(IplImage * output);	
	myColor myGet2D(int x, int y);
	void mySet2D(myColor colors, int x, int y);	
};

struct myFloatColor {
	double B; double G;  double R;
};

struct Obj_info {
public:
	int x;
	int y;
	int width;
	int height;
	int label;
	float distance;
	bool tracked;
	double Owner_R[10];
	double Owner_G[10];
	double Owner_B[10];
	int traj_label;
	float traj_dist;

	Obj_info() {
		label = 0;
		distance = 2147483647.0;
		tracked = false;
		traj_dist = 2147483647.0;
		for (int i = 0; i < 10; i++) {
			Owner_R[i] = 0.0; Owner_G[i] = 0.0; Owner_B[i] = 0.0;
			//for validation method 3 
		}
	}
};

struct pixelFSM {
	//short  long   state
	//  0    0        0
	//  0    1        1
	//  1    0        2
	//  1    1        3
public:
	int state_now;
	int state_pre;
	int static_count;
	bool staticFG_candidate;
	bool staticFG_stable;

	pixelFSM(){
		state_now = 0;
		state_pre = 0;
		static_count = 0;
		staticFG_candidate = false;
		staticFG_stable = false;
	}
};

struct gaussian
{
	double mean[3], covariance;
	double weight;	// Represents the measure to which a particular component defines the pixel value
	gaussian* Next;
	gaussian* Previous;

	gaussian() {
		mean[0] = 0.0; mean[1] = 0.0; mean[2] = 0.0; covariance = 0.0; weight = 0.0;
	};
};

struct Node
{
	gaussian* pixel_s;
	gaussian* pixel_r;
	int no_of_components;
	Node* Next;
};

class myGMM
{
public:
	gaussian  *ptr, *start, *rear, *g_temp, *save, *next, *previous, *nptr, *temp_ptr;
	Node  *N_ptr, *N_rear, *N_start;//, 

	//Some constants for the algorithm
	double pi;
	double cthr;
	double alpha;
	double cT;
	double covariance0;
	double cf;
	double cfbar;
	double temp_thr;
	double prune;
	double alpha_bar;
	//Temporary variable
	int overall;

	double del[3], mal_dist;
	double sum;
	double sum1;
	int count;
	bool close;
	int background;
	double mult;
	double duration, duration1, duration2, duration3;
	double temp_cov;
	double weight;
	double var;
	double muR, muG, muB, dR, dG, dB, rVal, gVal, bVal;

	unsigned char * r_ptr;
	unsigned char * b_ptr;

	//Some function associated with the structure management
	Node* Create_Node(double info1, double info2, double info3);
	void Insert_End_Node(Node* np);
	gaussian* Create_gaussian(double info1, double info2, double info3);
	void Insert_End_gaussian(gaussian* nptr);
	gaussian* Delete_gaussian(gaussian* nptr);

	//main function of GMM
	myGMM(double LearningRate);
	~myGMM();
	void process(myImage * inputRGB, myImage * outputBIN);
	void initial(myImage * inputRGB);
	void ChangeLearningRate(float new_learn_rate);

#ifdef USE_OPENCV
	void process(Mat orig_img, Mat bin_img);//this function is for openCV user
	void initial(Mat orig_img);//this function is for openCV user
#endif

private:
};

class CBM_model
{
public:
	int MOG_LEARN_FRAMES;
	int MIN_AREA;
	int TEMPORAL_RULE;

	CBM_model(myImage * input, int set_MOG_LearnFrame, int set_min_area,
		int set_buffer_len, float set_resize, myImage * mask);
	~CBM_model();
	void Initialize();
	bool Motion_Detection(myImage *img);

	bool myClustering(myImage *img, int option);//option: 0 for moving obj, 1 for static obj
	int GetLabeling(myImage *pImg1, int areaThreshold, int option); //option: 0 for moving obj, 1 for static obj

	void myFSM(myImage *short_term, myImage *long_term, pixelFSM ** imageFSM, bool *** Previous_FG);
	void myConvert2Img(bool **Array, myImage *output);
	void myCvtFSM2Img(pixelFSM **Array, myImage *Candidate_Fg, myImage *Static_Fg);

	int countFGnum(myImage * img);

	int frame_count;
	int sampling_idx;
	//bool singleFlag;

	//********RESIZE**************//
	int new_width;
	int new_height;
	float RESIZE_RATE;
	//********detected result**************//
	vector<Obj_info*> detected_result;//information of the moving objects
	vector<Obj_info*> static_object_result;//information of the static foreground objects	

	bool ** GetPrevious_nForeground(int n);
	cv::HOGDescriptor hog;//hog detector
	vector<cv::Rect>  found;

	myImage * _GetPrevious_nFrame(int n);

	void System_Reset();

	bool ***Previous_FG;

	myImage ** _Previous_Img;
	myImage * my_imgStatic;
	myImage * maskROI;
	myImage * input_temp;
private:
	myImage * my_mog_fg;//long term
	myImage * my_mog_fg2;//short term
	myImage * my_imgCandiStatic;
	
	myGMM * _myGMM;//long term
	myGMM * _myGMM2;//short term

	float MoG_LearningRate;// long term
	float MoG_LearningRate2;// short term
	int FG_count;

	pixelFSM **imageFSM;

	int staticFG_pixel_num_now;
	int staticFG_pixel_num_pre;
	int staticFG_pixel_num_pre2;

	myImage * dpm_gray;
	//parameters:
	int Win_width, Win_height, Bin_num,
		Block_size, Block_stride, Cell_size;
};

//=======================================
class ObjLeftDetect {
public:
	ObjLeftDetect(IplImage* mask);
	~ObjLeftDetect();
	bool process(IplImage* img, int fno);

	int** image;
	myFloatColor *connect_colors;
	CBM_model *_CBM_model;
	IplImage *B;
private:	
	int new_width, new_height;
	bool object_detected;
	bool set_alarm;

	vector<Obj_info*> ObjLocation;
	vector<Obj_info*> LeftLocation;
	vector<Obj_info*> alarmList;
	
	IplImage *imgTool;
	Mat mat_imgTool;	

	myImage *imgOri;	//original image
	myImage *mymask;	//mask
	myImage *_img;		//resized image

	myImage *_imgSynopsis;

	bool soft_validation3(myImage* ImgSynopsis, vector<Obj_info*> obj_left);
	int Spatial_Temporal_Search(int ** Image,int i, int j,
		myFloatColor * colors,int time_stamp,int my_label);
	int spatial_flood(bool ** foreground, int i, int j);
	myColor color_rainbow(int total_time, int current_time);
	void Set2DTool(IplImage* img, int i, int j, CvScalar pt);
	float point_dist(float x1, float y1, float x2, float y2);
};
