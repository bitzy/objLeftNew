#include "objLeftDetect.h"
#include <cstdio>
#define INPUT_RESIZE 0.5
#define GMM_LEARN_FRAME 300
#define MIN_FG 20
#define BUFFER_LENGTH 900
#define OWNER_SEARCH_ROI 50
#define MAX_FG 1000
#define MAX_SFG 120
#define MIN_SFG 10

template<class Tin, class Tlabel, class Comparator, class Boolean>
int ConnectedComponents::connected(const Tin *img, Tlabel *labelimg,
	int width, int height, Comparator SAME,Boolean K8_connectivity)
{	
	struct Label_handler {
		const Tin *piximg;
		Tlabel *labelimg;
		Label_handler(const Tin *img, Tlabel *limg) : piximg(img), labelimg(limg) {}
		Tlabel &operator()(const Tin *pixp) { return labelimg[pixp - piximg]; }
	} label(img, labelimg);
	
	clear();
	const Tin *row = img;
	const Tin *last_row = 0;
	label(&row[0]) = new_label();
	// label the first row.
	for (int c = 1, r = 0; c < width; ++c) {
		if (SAME(row[c], row[c - 1])) label(&row[c]) = label(&row[c - 1]);
		else label(&row[c]) = new_label();
	}
	// label subsequent rows.
	for (int r = 1; r < height; ++r)    {
		// label the first pixel on this row.
		last_row = row;
		row = &img[width*r];

		if (SAME(row[0], last_row[0])) label(&row[0]) = label(&last_row[0]);
		else label(&row[0]) = new_label();

		// label subsequent pixels on this row.
		for (int c = 1; c < width; ++c)	{
			int mylab = -1;

			// inherit label from pixel on the left if we're in the same blob.
			if (SAME(row[c], row[c - 1]))
				mylab = label(&row[c - 1]);
			for (int d = (K8_connectivity ? -1 : 0); d < 1; ++d) {
				// if we're in the same blob, inherit value from above pixel.
				// if we've already been assigned, merge its label with ours.
				if (SAME(row[c], last_row[c + d])) {
					if (mylab >= 0) merge(mylab, label(&last_row[c + d]));
					else mylab = label(&last_row[c + d]);
				}
			}
			if (mylab >= 0) label(&row[c]) = static_cast<Tlabel>(mylab);
			else label(&row[c]) = new_label();

			if (K8_connectivity && SAME(row[c - 1], last_row[c]))
				merge(label(&row[c - 1]), label(&last_row[c]));
		}
	}
	return relabel_image(labelimg, width, height);
}

template<class Tlabel>
int ConnectedComponents::relabel_image(Tlabel *labelimg, int width, int height)
{
	int newtag = 0;
	for (int id = 0; id < labels.size(); ++id)
		if (is_root_label(id))
			labels[id].tag = newtag++;

	for (int i = 0; i < width*height; ++i)
		labelimg[i] = labels[root_of(labelimg[i])].tag;

	return newtag;
}

//=============================================================================
void myImage::myImg_zero() {
#pragma omp parallel for 
	for (int i = 0; i < myw*myh*myd; i++)
		this->myData[i] = 0;
}
void myImage::myImg_inverse(myImage* output) {
	if ((myh != output->myh) || (myw != output->myw) || (myd != output->myd)){
		printf("[Error] myInverse:   height and width and depth cannot match\n");
		system("pause");
	}
	if (this->myd == 1) {
#pragma omp parallel for 
		for (int i = 0; i < myh; i++){
			for (int j = 0; j < myw; j++){
				if (myData[(j + i*myw)] == 255) output->myData[(j + i*myw)] = 0;
				else output->myData[(j + i*myw)] = 255;
			}
		}
	} else {
#pragma omp parallel for 
		for (int i = 0; i < myh; i++){
			for (int j = 0; j < myw; j++){
				int idx = j + i*myw;
				if (this->myData[idx] == 255) {
					output->myData[idx] = 0;
					output->myData[idx + myw*myh] = 0;
					output->myData[idx + (myw*myh<<1)] = 0;
				} else {
					output->myData[idx] = 255;
					output->myData[idx + myw*myh] = 255;
					output->myData[idx + (myw*myh<<1)] = 255;
				}
			}
		}
	}
}
void myImage::myImg_resize(myImage* output){	
	int scale1 = this->myh / output->myh;
	int scale2 = this->myw / output->myw;
	int w = output->myw;
	int h = output->myh;

	if (scale1 != scale2){
		printf("[ERROR] myResize:  different scaling parameter in width and height\n");
		system("pause");
	}
#pragma omp parallel for 
	for (int i = 0; i < output->myh; i++){
		for (int j = 0; j < output->myw; j++){
			output->myData[(j + i*w)] = this->myData[(j*scale1 + i*scale1*this->myw)];
			output->myData[(j + i*w) + (w*h * 1)] = this->myData[(j*scale1 + i*scale1*this->myw) + (this->myw*this->myh * 1)];
			output->myData[(j + i*w) + (w*h * 2)] = this->myData[(j*scale1 + i*scale1*this->myw) + (this->myw*this->myh * 2)];
		}
	}
}
void myImage::formMyImg(IplImage * input) {
	if ((input->width != this->myw) || (input->height != this->myh)) {
		printf("[Error] opencv_2_myImage: size not match!\n");
		system("pause");
	}	
#pragma omp parallel for 
	for (int i = 0; i < myw; i++){
		for (int j = 0; j < myh; j++){
			for (int k = 0; k < myd; k++){
				myData[(i + j*myw) + (myw*myh*k)] = 
					((uchar *)(input->imageData + j*input->widthStep))
					[i*input->nChannels + k];//B,G,R 
			}
		}
	}
}
void myImage::trans2IPL(IplImage * output) {	
	if ((this->myw != output->width) || (this->myh != output->height)){
		printf("[Error] myImage_2_opencv: size not match!\n");
		system("pause");
	}
#pragma omp parallel for 
for (int i = 0; i < myw; i++){
	for (int j = 0; j < myh; j++){
		for (int k = 0; k < myd; k++){
			((uchar *)(output->imageData + j*output->widthStep))[i*output->nChannels + k] = this->myData[(i + j*output->width) + (myw*myh*k)];//B,G,R 
		}
	}
}
}
myColor myImage::myGet2D(int x, int y) {		
	myColor colors;
	if (myd == 1) {
		colors.B = this->myData[(x + y*this->myw)];//B 
		colors.G = colors.B;//G 
		colors.R = colors.B;//R 
	} else if (myd == 3){
		colors.B = this->myData[(x + y*this->myw)];//B 
		colors.G = this->myData[(x + y*this->myw) + (myw*myh * 1)];//G 
		colors.R = this->myData[(x + y*this->myw) + (myw*myh * 2)];//R 
	}
	return colors;
}
void myImage::mySet2D(myColor colors, int x, int y) {		
	if (myd == 1) {
		this->myData[(x + y*this->myw)] = colors.B;//B 
	} else if (myd == 3) {
		this->myData[(x + y*this->myw)] = colors.B;//B 
		this->myData[(x + y*this->myw) + (myw*myh * 1)] = colors.G;//G 
		this->myData[(x + y*this->myw) + (myw*myh * 2)] = colors.R;//R 
	}
}
void myImage::myImg_and(myImage * input2, myImage * output) {	
	if ((this->myh != input2->myh) || (this->myw != input2->myw) ||
		(this->myw != output->myw) || (this->myw != output->myw)){
		printf("[Error] myImageAND:  height and width cannot match\n");
		system("pause");
	}
#pragma omp parallel for 
	for (int i = 0; i < myh; i++) {
		for (int j = 0; j < myw; j++) {
			if ((this->myData[(j + i*myw)] == 255) && 
				input2->myData[(j + i*myw)] == 255) {
				output->myData[(j + i*myw)] = 255;
			} else {
				output->myData[(j + i*myw)] = 0;
			}
		}
	}
}
void myImage::myImg_Dilation(int n) {
	myImage *temp = new myImage(myw, myh, myd);
#pragma omp parallel for 
	for (int r = 0; r < n; r++) {
		for (int j = 1; j < myh - 1; j++) {
			for (int i = 1; i < myw - 1; i++) {
				if (myData[j*myw + i] > 127) {
					for (int jj = j - 1; jj < j + 2; ++jj)
						for (int ii = i - 1; ii < i + 2; ++ii)
							temp->myData[jj * myw + ii] = 255;						
				}
			}
		}
		temp->myImg_copy(this);
	}
	temp->myReleaseImage();
	delete temp;
}
void myImage::myImg_copy(myImage * output){
	if ((this->myh != output->myh) || (this->myw != output->myw)){
		printf("cannot copy!   height and width cannot match\n");
		system("pause");
	}
#pragma omp parallel for 
	for (int i = 0; i < myw*myh*myd; i++)
		output->myData[i] = this->myData[i];

}
void myImage::myImg_rgb2gray(myImage * output) {
	if ((myh != output->myh) || (myw != output->myw)){
		printf("[Error] myRGB2Gray:   height and width cannot match\n");
		system("pause");
	}	
	if (output->myd == 1) {
#pragma omp parallel for 
		for (int i = 0; i < myh; i++){
			for (int j = 0; j < myw; j++){
				output->myData[(j + i*myw)] = (this->myData[(j + i*myw)] * 0.3) + 
					(this->myData[(j + i*myw) + (myw*myh)] * 0.3) + 
					(this->myData[(j + i*myw) + (myw*myh<<1)] * 0.3);
			}
		}
	} else  {
#pragma omp parallel for 
		for (int i = 0; i < myh; i++){
			for (int j = 0; j < myw; j++){
				int idx = j + i*myw;
				output->myData[idx] = myData[idx] * 0.3 + myData[idx + myw*myh] * 0.3 + 
					myData[idx + (myw*myh<<1)] * 0.3;
				output->myData[idx + myw*myh] = myData[idx] * 0.3 + myData[idx + myw*myh] * 0.3 + 
					myData[idx + (myw*myh<<1)] * 0.3;
				output->myData[idx + (myw*myh<<1)] = myData[idx] * 0.3 + myData[idx + myw*myh] * 0.3 + 
					myData[idx + (myw*myh<<1)] * 0.3;
			}
		}
	}
}

//=============================================================================
ObjLeftDetect::ObjLeftDetect(IplImage* mask) {	
	new_width = (int)(mask->width*INPUT_RESIZE);
	new_height = (int)(mask->height*INPUT_RESIZE);
	
	imgTool = cvCreateImage(cvSize(mask->width, mask->height), 8, 3);
	mat_imgTool = imgTool;//share data area;
	
	imgOri = new myImage(mask->width, mask->height, 3);	
	IplImage *mask_tmp = cvCreateImage(cvSize(new_width, new_height), IPL_DEPTH_8U, 1);
	cvResize(mask, mask_tmp);
	mymask = new myImage(new_width, new_height, 3);
	mymask->myImg_inverse(mymask);
	mymask->formMyImg(mask_tmp);
	cvReleaseImage(&mask_tmp);
	
	_CBM_model = new CBM_model(imgOri, GMM_LEARN_FRAME, 
		MIN_FG, BUFFER_LENGTH, INPUT_RESIZE, mymask);

	//initialize:
	object_detected = false;
	set_alarm = false;
	image = (int**)malloc(new_width*sizeof(int*));
	for (int k = 0; k < new_width; k++)
		image[k] = (int*)malloc(new_height*sizeof(int));
	connect_colors = (myFloatColor *)malloc(10*sizeof(myFloatColor));
	
	ObjLocation.clear();
	LeftLocation.clear();

	_img = new myImage(new_width, new_height, 3);
	_imgSynopsis = new myImage(new_width, new_height, 3);

	B = cvCreateImage(cvSize(new_width, new_height), 8, 3);
	cvZero(B);
	alarmList.clear();
}

ObjLeftDetect::~ObjLeftDetect() {
	cvReleaseImage(&imgTool);
	imgOri->myReleaseImage();
	delete imgOri;
	mymask->myReleaseImage();
	delete mymask;
	delete _CBM_model;
	for (int k = 0; k < new_width; k++)
		free(*(image+k));
	free(image);
	free(connect_colors);
	_img->myReleaseImage();
	delete _img;
	_imgSynopsis->myReleaseImage();
	delete _imgSynopsis;
	cvReleaseImage(&B);	
}

bool ObjLeftDetect::process(IplImage* dealFrame, int fno) {	
	cvCopy(dealFrame, imgTool);	
	medianBlur(mat_imgTool, mat_imgTool, 3);
	imgOri->formMyImg(imgTool); //imgTool ==> imgOri
	imgOri->myImg_resize(_img);
		
	set_alarm = false;	
	//static Foreground Detection
	object_detected = _CBM_model->Motion_Detection(_img);	
	if (object_detected == true) {
		ObjLocation = _CBM_model->detected_result;
		LeftLocation = _CBM_model->static_object_result;
		if (LeftLocation.size() > 0) {
			_imgSynopsis->myImg_zero();
			//back-Tracing verification && object left event analysis
			set_alarm = soft_validation3(_imgSynopsis, LeftLocation);
			_CBM_model->System_Reset();
			LeftLocation.clear();
		}
	}
	return set_alarm;
}

bool ObjLeftDetect::soft_validation3(myImage* ImgSynopsis, vector<Obj_info*> obj_left) 
{
	bool _set_alarm = false;
	bool ** foreground;
	int temporal_rule = BUFFER_LENGTH;
	int retreval_time = temporal_rule / 2 + temporal_rule / 6;
	/************************************************************************/
	/*  capture the color information of the suspected owner               */
	/************************************************************************/
	foreground = _CBM_model->GetPrevious_nForeground(retreval_time - 1);	
	bool foreground_found = false;
#pragma omp parallel for
	for (int j = 0; j < new_width; j++) {
		for (int k = 0; k < new_height; k++) {
			if (foreground[j][k] == true) {
				for (int n = 0; n < obj_left.size(); n++)
				{
					float owner_dist = point_dist((float)j, (float)k, (float)obj_left.at(n)->x, (float)obj_left.at(n)->y);
					if (owner_dist < OWNER_SEARCH_ROI)//distance threshold
					{
						for (int w = -30; w < 30; w++) {
							//rgb histogram accumulated
							myColor color;
							color = _CBM_model->_GetPrevious_nFrame(retreval_time - 1)->myGet2D(j, k);
							obj_left.at(n)->Owner_B[(int)((float)color.B / 255.0*10.0)] += 1.0;
							obj_left.at(n)->Owner_G[(int)((float)color.G / 255.0*10.0)] += 1.0;
							obj_left.at(n)->Owner_R[(int)((float)color.R / 255.0*10.0)] += 1.0;
						}
						foreground_found = true;
					}
				}
			}//if
		}
	}
	//if there is no foreground found near to the suspected left object, that is, we have a false alarm of object left detection
	//than return nothing!
	if (foreground_found == false) return false;
	else return true;
	
	////else if we found foreground object, it must be the owner, than we extract the color information of further processing 
	//myFloatColor ** obj_colors;
	//obj_colors = (myFloatColor **)malloc(obj_left.size()*sizeof(myFloatColor *));
	//for (int i = 0; i < obj_left.size(); i++)
	//	obj_colors[i] = (myFloatColor *)malloc(10 * sizeof(myFloatColor));
	//for (int i = 0; i < obj_left.size(); i++) {
	//	//normalizing the histogram accumulated information
	//	double total_r = 0.0, total_g = 0.0, total_b = 0.0;
	//	for (int j = 0; j < 10; j++) {
	//		total_r = total_r + obj_left.at(i)->Owner_R[j];
	//		total_g = total_g + obj_left.at(i)->Owner_G[j];
	//		total_b = total_b + obj_left.at(i)->Owner_B[j];
	//	}
	//	for (int j = 0; j < 10; j++) {
	//		obj_left.at(i)->Owner_R[j] = (obj_left.at(i)->Owner_R[j] / (total_r + 0.001));
	//		obj_left.at(i)->Owner_G[j] = (obj_left.at(i)->Owner_G[j] / (total_g + 0.001));
	//		obj_left.at(i)->Owner_B[j] = (obj_left.at(i)->Owner_B[j] / (total_b + 0.001));

	//		obj_colors[i][j].R = obj_left.at(i)->Owner_R[j];
	//		obj_colors[i][j].G = obj_left.at(i)->Owner_G[j];
	//		obj_colors[i][j].B = obj_left.at(i)->Owner_B[j];
	//	}
	//}
	//cvZero(B);
	///************************************************************************/
	///* use mounatin climbing algorithm to extract the candidate trajectory   */
	///************************************************************************/
	//int ** floodfillMap;
	//floodfillMap = (int **)malloc(new_width*sizeof(int *));
	//for (int i = 0; i < new_width; i++){
	//	floodfillMap[i] = (int *)malloc(new_height*sizeof(int));
	//}
	//for (int i = 0; i < new_width; i++){
	//	for (int j = 0; j < new_height; j++){
	//		floodfillMap[i][j] = 0;
	//	}
	//}
	//for (int n = 0; n < obj_left.size(); n++) {
	//	int _t_time = retreval_time - 1;
	//	obj_left.at(n)->traj_label = n + 1;

	//	for (int i = -OWNER_SEARCH_ROI / 2; i < OWNER_SEARCH_ROI / 2 + 1; i++) {
	//		for (int j = -OWNER_SEARCH_ROI / 2; j < OWNER_SEARCH_ROI / 2 + 1; j++) {
	//			Spatial_Temporal_Search(floodfillMap, 
	//				obj_left.at(n)->x + i, obj_left.at(n)->y + j, 
	//				obj_colors[n], _t_time, n + 1);
	//		}
	//	}
	//}

	///************************************************************************/
	///* Verifing the object left event by distance information              */
	///************************************************************************/	
	//for (int n = 0; n < obj_left.size(); n++){
	//	bool inside = false, outside = false;
	//	int inside_count = 0, outside_count = 0;
	//	for (int i = 0; i < new_width; i++){
	//		for (int j = 0; j < new_height; j++){
	//			//if (floodfillMap[i][j] == obj_left.at(n)->traj_label){
	//			if (floodfillMap[i][j] == (n + 1)) {
	//				float dist = point_dist((float)i, (float)j, (float)obj_left.at(n)->x, (float)obj_left.at(n)->y);
	//				if (dist > OWNER_SEARCH_ROI / 2) inside_count++;
	//				else if (dist < OWNER_SEARCH_ROI / 2) outside_count++;
	//			}
	//		}
	//	}
	//	
	//	if (inside_count > OWNER_SEARCH_ROI) inside = true;
	//	if (outside_count > OWNER_SEARCH_ROI) outside = true;
	//	if (inside && outside) _set_alarm = true;
	//}	
	//free(*floodfillMap);
	//free(floodfillMap);
	//free(*obj_colors);
	//free(obj_colors);
	//return _set_alarm;
}

int ObjLeftDetect::Spatial_Temporal_Search(int ** Image, int i, int j,
	myFloatColor * colors, int time_stamp, int my_label) {
	if ((j < 0) || (j >= new_height - 1)) return 0;
	if ((i < 0) || (i >= new_width - 1)) return 0;
	if (Image[i][j] == my_label) return 0;

	int object_x = i;
	int object_y = j;

	bool ** foreground;
	for (int t = time_stamp; t >= 2; t--)
	{
		foreground = _CBM_model->GetPrevious_nForeground(t);
		if (foreground[object_x][object_y] == true)
		{
			for (int k = 0; k < new_width; k++)
				for (int l = 0; l < new_height; l++)
					image[k][l] = 0;

			spatial_flood(foreground, object_x, object_y);

			for (int k = 0; k < 10; k++)
			{
				connect_colors[k].B = 0.0;
				connect_colors[k].G = 0.0;
				connect_colors[k].R = 0.0;
			}
			int min_x = new_width, max_x = 0;
			int min_y = new_height, max_y = 0;
			for (int k = 0; k < new_width; k++)
			{
				for (int l = 0; l < new_height; l++)
				{
					if (image[k][l] == 1)
					{
						myColor color_temp;
						color_temp = _CBM_model->_GetPrevious_nFrame(t)->myGet2D(k, l);
						connect_colors[(int)((double)color_temp.R / 255.0*10.0)].R += 1.0;
						connect_colors[(int)((double)color_temp.G / 255.0*10.0)].G += 1.0;
						connect_colors[(int)((double)color_temp.B / 255.0*10.0)].B += 1.0;

						if (min_x > k) min_x = k;
						if (max_x < k) max_x = k;

						if (min_y > l) min_y = l;
						if (max_y < l) max_y = l;

					}
				}
			}

			object_x = min_x + (max_x - min_x) / 2;
			object_y = min_y + (max_y - min_y) / 2;

			//normalizing the histogram accumulated information
			double total_r = 0.0, total_g = 0.0, total_b = 0.0;
			for (int w = 0; w < 10; w++)
			{
				total_r = total_r + connect_colors[w].R;
				total_g = total_g + connect_colors[w].G;
				total_b = total_b + connect_colors[w].B;
			}
			for (int w = 0; w < 10; w++)
			{
				connect_colors[w].R = connect_colors[w].R / (total_r + 0.001);
				connect_colors[w].G = connect_colors[w].G / (total_g + 0.001);
				connect_colors[w].B = connect_colors[w].B / (total_b + 0.001);
			}

			double color_prob_r = 0.0, color_prob_g = 0.0, color_prob_b = 0.0;
			for (int s = 0; s < 10; s++)
			{
				color_prob_r = color_prob_r + colors[s].R*connect_colors[s].R;
				color_prob_g = color_prob_g + colors[s].G*connect_colors[s].G;
				color_prob_b = color_prob_b + colors[s].B*connect_colors[s].B;
			}

			if ((color_prob_r >= 0.0) && (color_prob_g >= 0.0) && (color_prob_b >= 0.0))
			{
				int foreground_num = 0, overlap_num = 0;
				for (int k = 0; k < new_width; k++)
				{
					for (int l = 0; l < new_height; l++)
					{
						if ((Image[k][l] == my_label) && (image[k][l] == 1))
							overlap_num++;
					}
				}
				myColor color_temp, color_add;
				color_add = color_rainbow(time_stamp, t);
				for (int k = 0; k < new_width; k++)
				{
					for (int l = 0; l < new_height; l++)
					{
						if (image[k][l] == 1)
						{
							Image[k][l] = my_label;


							if (overlap_num < 1000)
							{
								color_temp = _CBM_model->_GetPrevious_nFrame(t)->myGet2D(k, l);
								CvScalar setcolor;
								setcolor.val[0] = color_temp.B + color_add.B;
								setcolor.val[1] = color_temp.G + color_add.G;
								setcolor.val[2] = color_temp.R + color_add.R;

								if (setcolor.val[0] > 255)	setcolor.val[0] = 255;
								if (setcolor.val[1] > 255)	setcolor.val[1] = 255;
								if (setcolor.val[2] > 255)	setcolor.val[2] = 255;

								Set2DTool(B, l, k, setcolor);
							}
						}
					}
				}
			}
		}
	}

	return 0;
}

int ObjLeftDetect::spatial_flood(bool ** foreground, int i, int j)
{
	if ((j < 0) || (j >= new_height - 1)) return 0;
	if ((i < 0) || (i >= new_width - 1)) return 0;
	if ((image[i][j] == 0) && (foreground[i][j] == true)){
		image[i][j] = 1;
		spatial_flood(foreground, i + 1, j);
		spatial_flood(foreground, i, j + 1);
		spatial_flood(foreground, i - 1, j);
		spatial_flood(foreground, i, j - 1);
	}
	return 0;
}

myColor ObjLeftDetect::color_rainbow(int total_time, int current_time)
{
	myColor color;
	double segment = (total_time + 0.001) / 6.0;
	int selection = (int)((double)current_time / segment);

	switch (selection){
	case 0: color.R = 100; color.G = 0; color.B = 200; break;
	case 1: color.R = 0; color.G = 0; color.B = 200; break;
	case 2: color.R = 0; color.G = 200; color.B = 0; break;
	case 3: color.R = 200; color.G = 200; color.B = 0; break;
	case 4: color.R = 200; color.G = 70; color.B = 0; break;
	case 5: color.R = 200; color.G = 0; color.B = 0; break;
	default: color.R = 0; color.G = 0; color.B = 0; break;
	}
	return color;
}

void ObjLeftDetect::Set2DTool(IplImage* img, int i, int j, CvScalar pt) {
	((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 0] = pt.val[0]; // B
	((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 1] = pt.val[1]; // G
	((uchar *)(img->imageData + i*img->widthStep))[j*img->nChannels + 2] = pt.val[2]; // R
}

float ObjLeftDetect::point_dist(float x1, float y1, float x2, float y2) {
	float tempX = (x1 - x2);
	float tempY = (y1 - y2);

	return sqrt(tempX*tempX + tempY*tempY);
}

//=============================================================================
CBM_model::CBM_model(myImage * input, int set_MOG_LearnFrame, int set_min_area,
	int set_buffer_len, float set_resize, myImage * mask)
{
	Win_width = 64;
	Win_height = 128;
	Block_size = 16;
	Block_stride = 8;
	Cell_size = 8;
	Bin_num = 9;

	frame_count = 0;
	sampling_idx = 0;
	FG_count = 0;
	//singleFlag = false;

	MOG_LEARN_FRAMES = set_MOG_LearnFrame;
	MIN_AREA = set_min_area;
	TEMPORAL_RULE = set_buffer_len;
	RESIZE_RATE = set_resize;

	new_width = (int)(input->myw*set_resize);
	new_height = (int)(input->myh*set_resize);

	Initialize();
	// Select parameters for Gaussian model.
	_myGMM = new myGMM(0.0001);//0.0001
	_myGMM2 = new myGMM(0.002);
	
	maskROI = mask;
	hog = HOGDescriptor(cvSize(Win_width, Win_height), cvSize(Block_size, Block_size), cvSize(Block_stride, Block_stride), cvSize(Cell_size, Cell_size), Bin_num);
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
}
CBM_model::~CBM_model()
{
	my_mog_fg->myReleaseImage();
	my_mog_fg2->myReleaseImage();
	my_imgCandiStatic->myReleaseImage();
	my_imgStatic->myReleaseImage();
	input_temp->myReleaseImage();
	for (int i = 0; i < new_width; i++)
		free(*(imageFSM + i));
	free(imageFSM);
	free(**Previous_FG);
	free(*Previous_FG);
	free(Previous_FG);
	for (int i = 0; i < TEMPORAL_RULE; i++){
		_Previous_Img[i]->myReleaseImage();
	}
}
void CBM_model::Initialize()
{	
	my_mog_fg = new myImage(new_width, new_height, 1);
	my_mog_fg2 = new myImage(new_width, new_height, 1);
	my_imgCandiStatic = new myImage(new_width, new_height, 3);
	my_imgStatic = new myImage(new_width, new_height, 3);	
	input_temp = new myImage(new_width, new_height, 1);
	imageFSM = (pixelFSM **)malloc((int)new_width*sizeof(pixelFSM *));
	for (int i = 0; i < new_width; i++){
		imageFSM[i] = (pixelFSM *)malloc((int)new_height*sizeof(pixelFSM));
	}
	Previous_FG = (bool ***)malloc(TEMPORAL_RULE*sizeof(bool **));
	for (int i = 0; i < TEMPORAL_RULE; i++){
		Previous_FG[i] = (bool **)malloc((int)new_width*sizeof(bool *));
	}
	for (int i = 0; i < TEMPORAL_RULE; i++){
		for (int j = 0; j < new_width; j++){
			Previous_FG[i][j] = (bool *)malloc((int)new_height*sizeof(bool));
		}
	}
	_Previous_Img = (myImage **)malloc(TEMPORAL_RULE*sizeof(myImage *));
	for (int i = 0; i < TEMPORAL_RULE; i++){
		_Previous_Img[i] = new myImage(new_width, new_height, 3);
	}
	staticFG_pixel_num_now = -1;
	staticFG_pixel_num_pre = -2;
	staticFG_pixel_num_pre2 = -3;

	dpm_gray = new myImage(new_width, new_height, 1);
	printf("CBM initialize done!\n");
}
void CBM_model::System_Reset()
{
#pragma omp parallel for
	for (int i = 0; i < new_width; i++){
		for (int j = 0; j < new_height; j++){
			imageFSM[i][j].state_now = 0;
			imageFSM[i][j].staticFG_stable = false;
			imageFSM[i][j].staticFG_candidate = false;
			imageFSM[i][j].static_count = 0;
		}
	}
	static_object_result.clear();
}
bool CBM_model::Motion_Detection(myImage *img)
{
	img->myImg_resize(_Previous_Img[FG_count]);
	if (frame_count < MOG_LEARN_FRAMES){
		printf("update mog %d\n", MOG_LEARN_FRAMES-frame_count);
		if (frame_count == 0){
			_myGMM->initial(_Previous_Img[FG_count]);
			_myGMM2->initial(_Previous_Img[FG_count]);
		}
		_myGMM->process(_Previous_Img[FG_count], my_mog_fg);
		_myGMM2->process(_Previous_Img[FG_count], my_mog_fg2);
		frame_count++;
		return false;
	} else {
		/*if (singleFlag == false) {
			singleFlag = true;
			return false;
		}
		singleFlag = true;*/
		//***MOG model***//
		//long:
		_myGMM->process(_Previous_Img[FG_count], input_temp);
		input_temp->myImg_and(maskROI, my_mog_fg);
		my_mog_fg->myImg_Dilation(3);
		//short:
		_myGMM2->process(_Previous_Img[FG_count], input_temp);
		input_temp->myImg_and(maskROI, my_mog_fg2);		
		my_mog_fg2->myImg_Dilation(3);
		//adjust long term _myGMM learn rate:
		if (countFGnum(my_mog_fg) > (my_mog_fg->myw*my_mog_fg->myh*0.30)){
			//if motion detection cannot work well		
			//speed up long-term model's learning rate to adapt the lighting changes.
			_myGMM->ChangeLearningRate(0.02);
		} else {//default long-term model learning rate
			_myGMM->ChangeLearningRate(0.0001);
		}
		myFSM(my_mog_fg2, my_mog_fg, imageFSM, Previous_FG);
		myCvtFSM2Img(imageFSM, my_imgCandiStatic, my_imgStatic);

		staticFG_pixel_num_pre2 = staticFG_pixel_num_pre;
		staticFG_pixel_num_pre = staticFG_pixel_num_now;
		staticFG_pixel_num_now = countFGnum(my_imgStatic);	
		bool static_object_detected = false;
		if ((staticFG_pixel_num_now == staticFG_pixel_num_pre) && 
			(staticFG_pixel_num_pre == staticFG_pixel_num_pre2) && 
			(staticFG_pixel_num_now > 0)) {
			static_object_detected = myClustering(my_imgStatic, 1);
		}
		FG_count = FG_count + 1;
		FG_count = FG_count%TEMPORAL_RULE;
		
		return static_object_detected;
	}
}
bool CBM_model::myClustering(myImage *img, int option)
{
	int area_threshold = 0;
	myImage * temp = new myImage(new_width, new_height, 1);
	if (img->myd == 3)//static foreground object
	{
		img->myImg_rgb2gray(temp);
		area_threshold = MIN_AREA / 2;//0;
	}
	else if (img->myd == 1)//foreground detection
	{
		img->myImg_copy(temp);
		area_threshold = MIN_AREA;
	}

	int found_objnum = 0;
	found_objnum = GetLabeling(temp, area_threshold, option);
	temp->myReleaseImage();
	delete(temp);

	if (found_objnum > 0)  return true;
	else return false;
}
/************************************************************************/
/*
GetLabeling : input a binary frame, bounding the connected component.
Ignore the connected component when :  
case1.  It's pixel is more than a areaThreshold.
case2.  The bounding rectangle is too thin or fat.  */
/************************************************************************/
int CBM_model::GetLabeling(myImage *pImg1, int areaThreshold, int option)
{
	int	found_objnum = 0;
	if (option == 0) detected_result.clear();
	else if (option == 1) static_object_result.clear();

	//find object's conturs of binary frame 
	unsigned int *out = (unsigned int *)malloc(sizeof(*out)*pImg1->myw*pImg1->myh);
	for (int i = 0; i < pImg1->myw*pImg1->myh; i++){
		out[i] = pImg1->myData[i];
	}

	ConnectedComponents cc(30);
	cc.connected(pImg1->myData, out, pImg1->myw, pImg1->myh,
		std::equal_to<unsigned char>(), constant<bool, true>());	

	bool constant_template[256] = { false };
	vector<int> color_labels;
	color_labels.clear();
	for (int i = 0; i < pImg1->myw*pImg1->myh; i++){
		constant_template[out[i]] = true;
	}
	for (int i = 0; i < 256; i++){
		if (constant_template[i] == true){
			found_objnum++;
			color_labels.push_back(i);
		}
	}
	if (found_objnum == 1){
		free(out);
		return found_objnum - 1;
	}
	else{
		for (int n = 0; n < found_objnum; n++)
		{
			int blob_x1 = pImg1->myw, blob_y1 = pImg1->myh, blob_x2 = 0, blob_y2 = 0;
			for (int i = 0; i < pImg1->myw; i++){
				for (int j = 0; j < pImg1->myh; j++){
					if (out[i + j*pImg1->myw] == color_labels.at(n)){
						if (i < blob_x1)  blob_x1 = i;
						if (j < blob_y1)  blob_y1 = j;
						if (i > blob_x2)  blob_x2 = i;
						if (j > blob_y2)  blob_y2 = j;
					}
				}
			}
			int blob_w = 0, blob_h = 0;
			blob_w = (blob_x2 - blob_x1) + 1;
			blob_h = (blob_y2 - blob_y1) + 1;

			//rectangle ratio filter
			int areaThreshold_max = 0, areaThreshold_min = 0;
			if (option == 0)//for moving foreground
			{
				areaThreshold_max = MAX_FG;
				areaThreshold_min = MIN_FG;
			}
			else if (option == 1)
			{
				areaThreshold_max = MAX_SFG;
				areaThreshold_min = MIN_SFG;
			}

			if ((((int)blob_w*(int)blob_h) > areaThreshold_min) &&
				(((int)blob_w*(int)blob_h) < (float)areaThreshold_max))
			{
				Obj_info * element;
				element = new Obj_info;
				element->x = blob_x1 + blob_w / 2;
				element->y = blob_y1 + blob_h / 2;
				element->width = blob_w;
				element->height = blob_h;
				//cvRectangle( img, cvPoint(blob_x1,blob_y1), cvPoint(blob_x2,blob_y2), CV_RGB(255,255,255), 2, 8, 0);

				if (option == 0)	detected_result.push_back(element);
				if (option == 1)	static_object_result.push_back(element);
			}//end of filter  
		}
		free(out);
		return found_objnum - 1;
	}//end of object checking
}
myImage * CBM_model::_GetPrevious_nFrame(int n) {
	return _Previous_Img[(FG_count + (TEMPORAL_RULE - n)) % TEMPORAL_RULE];
}
bool ** CBM_model::GetPrevious_nForeground(int n) {
	return Previous_FG[(FG_count + (TEMPORAL_RULE - n)) % TEMPORAL_RULE];
}
void CBM_model::myFSM(myImage *short_term, myImage *long_term, pixelFSM ** imageFSM, bool *** Previous_FG)
{
	myColor buffer[2];
#pragma omp parallel for
	for (int i = 0; i < new_width; i++){
		for (int j = 0; j < new_height; j++){
			buffer[0] = short_term->myGet2D(i, j);
			buffer[1] = long_term->myGet2D(i, j);

			imageFSM[i][j].state_pre = imageFSM[i][j].state_now;
			imageFSM[i][j].state_now = 0;

			if ((buffer[0].B == 255) && (buffer[0].G == 255) && (buffer[0].R == 255)){
				imageFSM[i][j].state_now += 2;
			}
			else{
				imageFSM[i][j].state_now = 0;
			}

			if ((buffer[1].B == 255) && (buffer[1].G == 255) && (buffer[1].R == 255)){
				imageFSM[i][j].state_now++;
			}
			else{
				imageFSM[i][j].state_now = 0;
			}

			if ((imageFSM[i][j].state_now == 1) && (imageFSM[i][j].state_pre == 1)){
				if (imageFSM[i][j].static_count == (TEMPORAL_RULE / 2)){
					imageFSM[i][j].staticFG_stable = true;
				}

				if (imageFSM[i][j].staticFG_candidate == true){
					imageFSM[i][j].static_count++;
				}
			}
			else
			{
				imageFSM[i][j].static_count = 0;
				imageFSM[i][j].staticFG_candidate = false;
			}

			if ((imageFSM[i][j].state_now == 1) && (imageFSM[i][j].state_pre == 3))
			{
				imageFSM[i][j].staticFG_candidate = true;
			}

			if (imageFSM[i][j].state_now == 3)
				Previous_FG[FG_count][i][j] = true;
			else
				Previous_FG[FG_count][i][j] = false;
		}
	}
}
void CBM_model::myConvert2Img(bool **Array, myImage *output)
{
#pragma omp parallel for
	for (int i = 0; i < new_width; i++){
		for (int j = 0; j < new_height; j++){
			if (Array[i][j] == true){
				myColor a; a.B = 255; a.G = 0; a.R = 0;
				output->mySet2D(a, i, j);
			}
			else{
				myColor a; a.B = 0; a.G = 0; a.R = 0;
				output->mySet2D(a, i, j);
			}
		}
	}
}
void CBM_model::myCvtFSM2Img(pixelFSM **Array, myImage *Candidate_Fg, myImage *Static_Fg)
{
	myColor color1, color2;
	color1.B = 0; color1.G = 0; color1.R = 255;
	color2.B = 0; color2.G = 200; color2.R = 255;
#pragma omp parallel for	
	for (int i = 0; i < new_width; i++){
		for (int j = 0; j < new_height; j++){
			if (Array[i][j].staticFG_candidate == true)
				Candidate_Fg->mySet2D(color1, i, j);
			else{
				myColor a; a.B = 0; a.G = 0; a.R = 0;
				Candidate_Fg->mySet2D(a, i, j);
			}

			if (Array[i][j].staticFG_stable == true)
				Static_Fg->mySet2D(color2, i, j);
			else{
				myColor a; a.B = 0; a.G = 0; a.R = 0;
				Static_Fg->mySet2D(a, i, j);
			}
		}
	}
}
int CBM_model::countFGnum(myImage * img) {
	int foregroud = 0;
	myColor a;
	for (int i = 0; i < img->myw; i++) {
		for (int j = 0; j < img->myh; j++) {
			a = img->myGet2D(i, j);
			if ((a.B >= 100) || (a.G >= 100) || (a.R >= 100)) {
				foregroud++;
			}
		}
	}
	return foregroud;
}
