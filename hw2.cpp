#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include<cmath>
#include<map>
#include<queue>

using namespace cv;
using namespace std;
vector<vector<Point>> att_borders;
vector<vector<Point>> rest_borders;
map<int, string> shape_Type;
multimap<int, string> attach_Type;

string filename;
vector<vector<Point>>Blob;

void attach_print(Mat img, vector<vector<Point>> borders) {
	Mat attach_img = Mat(img.size(), CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < borders.size(); i++) {
		vector<Point> v = borders[i];
		for (int j = 0; j < v.size(); j++) {
			attach_img.at<Vec3b>(v[j])[0] = 0;
			attach_img.at<Vec3b>(v[j])[1] = 0;
			attach_img.at<Vec3b>(v[j])[2] = 0;
		}
	}

	for (int i = 0; i < att_borders.size(); i++) {
		vector<Point> v = att_borders[i];
		int r = 100, g = 100, b = 100;
		if (i % 5 == 0) { r = 255; b = 0; g = 0; }
		if (i % 5 == 1) { b = 255; r = 0; g = 0; }
		if (i % 5 == 2) { r = 0; b = 0; g = 255; }
		if (i % 5 == 3) { r = 255; b = 0; g = 255; }
		if (i % 5 == 4) { r = 0; b = 255; g = 255; }
		for (int j = 0; j < v.size(); j++) {
			attach_img.at<Vec3b>(v[j])[0] = b;
			attach_img.at<Vec3b>(v[j])[1] = g;
			attach_img.at<Vec3b>(v[j])[2] = r;
		}
	}

	//imshow("black", attach_img);
	imwrite("image_p3_" + filename  + ".png", attach_img);
}
String check(Point top, Point bottom, Point left, Point right, vector<Point> v) {
	int t = top.x, b = bottom.x, l = left.y, r = right.y;
	int triangle_flag = 0;
	for (int i = 0; i < v.size(); i++) {
		//top-left and top-right
		if (abs(v[i].y - l) < 2 && abs(v[i].x - t) < 2) {
			//cout<<"left";
			return "square";
		}
		if ((abs(v[i].y - r) < 2 && abs(v[i].x - t) < 2)) {
			//cout<<"right";

			return "square";
		}
		//bottom-left and bottom-right
		if (abs(v[i].x - b) < 3 && abs(v[i].y - l) < 2) { triangle_flag++; }
		if (abs(v[i].x - b) < 3 && abs(v[i].y - r) < 2) { triangle_flag++; }
		if (triangle_flag >= 2)return "triangle";

		int x1 = top.x - left.x; int y1 = top.y - left.y;
		int x2 = top.x - right.x; int y2 = top.y - right.y;
		double a_1 = atan2(x1, y1) * 180 / 3.1415926;
		double a_2 = atan2(x2, y2) * 180 / 3.1415926;
		double a = abs(a_1 - a_2);
		if (a > 55 && a < 65) {
			return "triangle";
		}

	}
	return "circle";
}
void p5_shapeClassification(vector<vector<Point>> borders, Mat res_img) {
	//rest_borders
	rest_borders = borders;
	multimap<int, string> shape_Type;
	for (int i = 0; i < rest_borders.size(); i++) {
		for (int k = 0; k < att_borders[i].size(); k++) {
			for (int j = 0, f = 0; j < rest_borders[i].size(); j++) {
				if (rest_borders[i][j] == att_borders[i][k]) {
					//if(i==1){cout<<att_borders[i][k]<<endl;}
					vector<Point>::iterator it = rest_borders[i].begin();
					rest_borders[i].erase(it + j - f);
					f++;
				}
			}
		}
	}

	for (int i = 0; i < rest_borders.size(); i++) {
		vector<Point> b = rest_borders[i];
		Point top = Point(res_img.rows, 0), left = Point(0, res_img.cols), bottom = Point(-1, -1), right = Point(-1, -1);
		for (int j = 0; j < b.size(); j++) {
			if (b[j].x < top.x) {
				top = b[j];
			}
			if (b[j].x > bottom.x) {
				bottom = b[j];
			}
			if (b[j].y < left.y) {
				left = b[j];
			}
			if (b[j].y > right.y) {
				right = b[j];
			}
		}

		Point tp = b[0];

		shape_Type.insert(make_pair(res_img.at<uchar>(tp), check(top, bottom, left, right, b)));
		cout << endl << endl;
	}

	map<int, string>::iterator iter = shape_Type.begin();

	for (; iter != shape_Type.end(); iter++) {
		cout << iter->second << endl;
	}
}

void p3_objectAgainst(Mat img, Mat res_img, vector<vector<Point>> borders) {
	vector<Point> attach;


	multimap<int, string> shape_Type;

	for (int i = 0; i < borders.size(); i++) {
		att_borders.push_back(attach);
	}

	//add img_borders
	vector<Point>img_border;
	for (int i = 0; i < img.rows; i++) {
		if (i == 0 || i == img.rows - 1) {
			for (int j = 0; j < img.cols; j++) {
				img_border.push_back(Point(i, j));
			}
		}
		else {
			img_border.push_back(Point(i, 0));
			img_border.push_back(Point(i, img.cols - 1));
		}
	}
	borders.push_back(img_border);


	cout << endl;
	//vector<Point> bd1=borders[0]; vector<Point> bd2=borders[1];
	for (int m = 0; m < borders.size() - 1; m++) {
		vector<Point> bd1 = borders[m];
		vector<Point> att1;
		// cout<<endl<<"border"<<m+1<<endl;
		for (int n = m + 1; n < borders.size(); n++) {
			vector<Point> bd2 = borders[n];
			vector<Point> att2;
			int sum_x = 0, sum_y = 0, count = 0;
			//find attachment
			for (int i = 0, f1 = 0; i < bd1.size(); i++) {
				int flag = 0;
				for (int j = 0, f2 = 0; j < bd2.size(); j++) {

					// if attached
					if (abs(bd1[i].x - bd2[j].x) < 2 && abs(bd1[i].y - bd2[j].y) < 2) {
						//cout<<bd1[i]<<"->"<<bd2[j];
						sum_x += (bd1[i].x + bd2[j].x);
						sum_y += (bd1[i].y + bd2[j].y);
						count += 2;
						if (flag < 1) {
							att_borders[m].push_back(Point(bd1[i].x, bd1[i].y));

						}
						if (n < borders.size() - 1)
							att_borders[n].push_back(Point(bd2[j].x, bd2[j].y));
						// cout<<"  att2:"<<att_borders[n].size()<<endl;



						flag++;

					}
				}
			}

			if (count == 0) { count = -1; };
			//cout<<"avg:"<<Point(sum_x/count,sum_y/count)<<endl<<endl;

			if (n == borders.size() - 1) {
				//background block

				if (count > 0 && (sum_x / count - 2 < 0 || (sum_x / count + 2) > (res_img.rows - 1) || sum_y / count - 2 < 0 || (sum_x / count + 2) > (res_img.cols - 1))) {
					cout << "object" << (int)res_img.at<uchar>(bd1[0].x, bd1[0].y) << "againt the background" << endl;
					attach_Type.insert(make_pair((int)res_img.at<uchar>(bd1[0].x, bd1[0].y), "against the background"));

				}

				else {
					//whole?
					if (attach_Type.count((int)res_img.at<uchar>(bd1[0].x, bd1[0].y)) == 0) {

						attach_Type.insert(make_pair((int)res_img.at<uchar>(bd1[0].x, bd1[0].y), "whole image"));
						cout << "blob" << (int)res_img.at<uchar>(bd1[0].x, bd1[0].y) << " the whole image" << endl;
					}
				}
			}
			//blob block?
			else {
				if (res_img.at<uchar>(sum_x / count, sum_y / count) == res_img.at<uchar>(bd2[0].x, bd2[0].y)) {

					cout << "blob" << (int)res_img.at<uchar>(bd1[0].x, bd1[0].y) << "is blocked by blob" << (int)res_img.at<uchar>(bd2[1].x, bd2[1].y) << endl;
					attach_Type.insert(make_pair((int)res_img.at<uchar>(bd1[0].x, bd1[0].y), "against another shape blob"));

				}
				else {
					cout << "blob" << (int)res_img.at<uchar>(bd2[0].x, bd2[0].y) << "is blocked by blob" << (int)res_img.at<uchar>(bd1[0].x, bd1[0].y) << endl;
					attach_Type.insert(make_pair((int)res_img.at<uchar>(bd2[0].x, bd2[0].y), "against another shape blob"));
				}

			}
		}
	}

	multimap<int, int>::const_iterator _it;
	multimap<int, string>::const_iterator _its;

	cout << endl;
}

Point findFirstB(Point c, int rows, int cols) {
	Point b;
	int x = c.x;
	int y = c.y;
	if (y - 1 < 0) {      //west invaild
		if (x - 1 < 0) { //north invaild
			if (y + 1 >= cols) { //east invaild
				if (x + 1 >= cols) {
					return Point(0, 0);
				}
				else
					return Point(x + 1, y);
			}
			else
				return Point(x, y + 1);
		}
		else
			return Point(x - 1, y);
	}
	else
		return Point(x, y - 1);
}
Point findFirstC(Mat binary_img, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (binary_img.at<uchar>(i, j) == 255)
				return Point(i, j);
		}
	}
}

vector<vector<Point>>  findContours_diy(Mat binary_img) {
	Point first_c, pre_b;
	Point curr_c, curr_b;
	vector<vector<Point>> contours_diy;
	vector<Point> contour;
	Mat binary_img_extend;
	copyMakeBorder(binary_img, binary_img_extend, 1, 1, 1, 1, BORDER_REFLECT_101, 0);
	int rows = binary_img_extend.rows;
	int cols = binary_img_extend.cols;
	for (int i = 0; i < rows; i++) {
		binary_img_extend.at<uchar>(0, i) = 0;
		binary_img_extend.at<uchar>(rows - 1, i) = 0;
	}
	for (int i = 0; i < cols; i++) {
		binary_img_extend.at<uchar>(i, 0) = 0;
		binary_img_extend.at<uchar>(i, cols - 1) = 0;
	}
	int count = 0;
	bool flag = true;
	curr_c = findFirstC(binary_img_extend, rows, cols);
	first_c = curr_c;
	curr_b = findFirstB(curr_c, rows, cols);
	while (first_c != curr_c || flag == true) {
		count++;
		if (curr_b.y - curr_c.y == -1 && curr_b.x == curr_c.x) {  //b is on the left side of c
			if (binary_img_extend.at<uchar>(curr_b.x, curr_b.y) == 255) {    //find new one The current b is set to c, and b becomes pre_b
				flag = false;
				curr_c.x = curr_b.x;
				curr_c.y = curr_b.y;
				//contour.push_back(Point(curr_c.x, curr_c.y));
				contour.push_back(Point(curr_c.x - 1, curr_c.y - 1));
				curr_b.x = pre_b.x;
				curr_b.y = pre_b.y;
			}
			else {
				pre_b.x = curr_b.x;
				pre_b.y = curr_b.y;
				curr_b.x = curr_b.x - 1;  //b becomes top left
				curr_b.y = curr_b.y;
			}
		}
		else if (curr_b.y - curr_c.y == -1 && curr_b.x - curr_c.x == -1) { //b is on the left top of c
			if (binary_img_extend.at<uchar>(curr_b.x, curr_b.y) == 255) { //find new one The current b is set to c, and b becomes pre_b
				flag = false;
				curr_c.x = curr_b.x;
				curr_c.y = curr_b.y;
				//contour.push_back(Point(curr_c.x, curr_c.y));
				contour.push_back(Point(curr_c.x - 1, curr_c.y - 1));
				curr_b.x = pre_b.x;
				curr_b.y = pre_b.y;
			}
			else {
				pre_b.x = curr_b.x;
				pre_b.y = curr_b.y;
				curr_b.x = curr_b.x;  //b becomes top 
				curr_b.y = curr_b.y + 1;
			}
		}
		else if (curr_b.y - curr_c.y == -1 && curr_b.x - curr_c.x == 1) { //b is on the left bottom of c
			if (binary_img_extend.at<uchar>(curr_b.x, curr_b.y) == 255) { //find new one The current b is set to c, and b becomes pre_b
				flag = false;
				curr_c.x = curr_b.x;
				curr_c.y = curr_b.y;
				//contour.push_back(Point(curr_c.x, curr_c.y));
				contour.push_back(Point(curr_c.x - 1, curr_c.y - 1));
				curr_b.x = pre_b.x;
				curr_b.y = pre_b.y;
			}
			else {
				pre_b.x = curr_b.x;
				pre_b.y = curr_b.y;
				curr_b.x = curr_b.x - 1;  //b becomes left side
				curr_b.y = curr_b.y;
			}
		}
		else if (curr_b.y == curr_c.y && curr_b.x - curr_c.x == -1) { //b is on the top of c
			if (binary_img_extend.at<uchar>(curr_b.x, curr_b.y) == 255) { //find new one The current b is set to c, and b becomes pre_b
				flag = false;
				curr_c.x = curr_b.x;
				curr_c.y = curr_b.y;
				//contour.push_back(Point(curr_c.x, curr_c.y));
				contour.push_back(Point(curr_c.x - 1, curr_c.y - 1));
				curr_b.x = pre_b.x;
				curr_b.y = pre_b.y;
			}
			else {
				pre_b.x = curr_b.x;
				pre_b.y = curr_b.y;
				curr_b.x = curr_b.x;  //b becomes top right
				curr_b.y = curr_b.y + 1;
			}
		}
		else if (curr_b.y == curr_c.y && curr_b.x - curr_c.x == 1) { //b is on the bottom of c
			if (binary_img_extend.at<uchar>(curr_b.x, curr_b.y) == 255) { //find new one The current b is set to c, and b becomes pre_b
				curr_c.x = curr_b.x;
				curr_c.y = curr_b.y;
				//contour.push_back(Point(curr_c.x, curr_c.y));
				contour.push_back(Point(curr_c.x - 1, curr_c.y - 1));
				curr_b.x = pre_b.x;
				curr_b.y = pre_b.y;
			}
			else {
				pre_b.x = curr_b.x;
				pre_b.y = curr_b.y;
				curr_b.x = curr_b.x;  //b becomes left bottom
				curr_b.y = curr_b.y - 1;
			}
		}
		else if (curr_b.y - curr_c.y == 1 && curr_b.x == curr_c.x) { //b is on the right side of c
			if (binary_img_extend.at<uchar>(curr_b.x, curr_b.y) == 255) { //find new one The current b is set to c, and b becomes pre_b
				flag = false;
				curr_c.x = curr_b.x;
				curr_c.y = curr_b.y;
				//contour.push_back(Point(curr_c.x, curr_c.y));
				contour.push_back(Point(curr_c.x - 1, curr_c.y - 1));
				curr_b.x = pre_b.x;
				curr_b.y = pre_b.y;
			}
			else {
				pre_b.x = curr_b.x;
				pre_b.y = curr_b.y;
				curr_b.x = curr_b.x + 1;  //b becomes right bottom
				curr_b.y = curr_b.y;
			}
		}
		else if (curr_b.y - curr_c.y == 1 && curr_b.x - curr_c.x == -1) { //b is on the right top of c
			if (binary_img_extend.at<uchar>(curr_b.x, curr_b.y) == 255) { //find new one The current b is set to c, and b becomes pre_b
				flag = false;
				curr_c.x = curr_b.x;
				curr_c.y = curr_b.y;
				//contour.push_back(Point(curr_c.x, curr_c.y));
				contour.push_back(Point(curr_c.x - 1, curr_c.y - 1));
				curr_b.x = pre_b.x;
				curr_b.y = pre_b.y;
			}
			else {
				pre_b.x = curr_b.x;
				pre_b.y = curr_b.y;
				curr_b.x = curr_b.x + 1;  //b becomes right side
				curr_b.y = curr_b.y;
			}
		}
		else if (curr_b.y - curr_c.y == 1 && curr_b.x - curr_c.x == 1) { //b is on the right bottom of c
			if (binary_img_extend.at<uchar>(curr_b.x, curr_b.y) == 255) { //find new one The current b is set to c, and b becomes pre_b
				flag = false;
				curr_c.x = curr_b.x;
				curr_c.y = curr_b.y;
				//contour.push_back(Point(curr_c.x, curr_c.y));
				contour.push_back(Point(curr_c.x - 1, curr_c.y - 1));
				curr_b.x = pre_b.x;
				curr_b.y = pre_b.y;
			}
			else {
				pre_b.x = curr_b.x;
				pre_b.y = curr_b.y;
				curr_b.x = curr_b.x;  //b becomes bottom
				curr_b.y = curr_b.y - 1;
			}
		}
	}
	cout << "diy loop : " << count << endl;
	contours_diy.push_back(contour);
	return contours_diy;
}

vector<vector<Point>> p2_diy(Mat img, Mat res_img) {
	vector<vector<Point>> contours;
	vector<vector<Point>> diy_contours;
	vector<Point> contour_diy;
	vector<Vec4i> hierarchy;
	vector<Mat> binary_img;
	vector<Mat> contour_diy_img;

	int maxsize = 0;
	int maxind = 0;
	Rect boundrec;
	int rows = img.rows;
	int cols = img.cols;
	int x, y;
	int number;
	for (int i = 0; i < Blob.size(); i++) {
		if (Blob[i].size() > 0) {
			number = res_img.at<uchar>(Blob[i][0].x, Blob[i][0].y);
			Mat temp_img = res_img.clone();
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (res_img.at<uchar>(i, j) == number)
						temp_img.at<uchar>(i, j) = 255;
					else
						temp_img.at<uchar>(i, j) = 0;
				}
			}
			binary_img.push_back(temp_img);
		}
	}
	cout << "binary image size is (diy)" << binary_img.size() << endl;
	for (int i = 0; i < binary_img.size(); i++) {
		contours = findContours_diy(binary_img[i]);
		Mat contour_output = Mat::zeros(binary_img[i].size(), CV_8UC3);
		cout << "The number of contours detected is: (diy)" << contours.size() << endl;
		diy_contours.push_back(contours[0]);
		if (contours.size() > 0) {
			for (int i = 0; i < contours.size(); i++)
			{
				double area = contourArea(contours[i]);
				if (area > maxsize) {
					maxsize = area;
					maxind = i;
					boundrec = boundingRect(contours[i]);
				}
			}
			//drawContours(contour_output, contours, -1, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
			drawContours(contour_output, contours, -1, Scalar(0, 255, 0), 2, 8, hierarchy);

			contour_diy_img.push_back(contour_output);
		}
	}
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(4);
	for (int i = 0; i < contour_diy_img.size(); i++) {
		string num = std::to_string(i);
		string s = "diy_contour" + num;
		//imshow(s, contour_diy_img[i]);
		imwrite("image_" + filename + "_p2_diy" + num + ".png", contour_diy_img[i]);

	}
	return diy_contours;
}

vector<vector<Point>> p2_opencv(Mat img, Mat res_img) {

	vector<vector<Point>> contours;
	vector<vector<Point>> opencv_contours;
	vector<Vec4i> hierarchy;
	vector<Mat> binary_img;
	vector<Mat> contour_img;

	int maxsize = 0;
	int maxind = 0;
	Rect boundrec;
	int rows = img.rows;
	int cols = img.cols;
	int x, y;
	int number;
	for (int i = 0; i < Blob.size(); i++) {  //for each object in the img, gennerate a binary img for this object
		if (Blob[i].size() > 0) {
			number = res_img.at<uchar>(Blob[i][0].x, Blob[i][0].y);
			Mat temp_img = res_img.clone();
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (res_img.at<uchar>(i, j) == number)
						temp_img.at<uchar>(j, i) = 255;
					else
						temp_img.at<uchar>(j, i) = 0;
				}
			}
			binary_img.push_back(temp_img);
		}
	}
	cout << "binary image size is " << binary_img.size() << endl;


	for (int i = 0; i < binary_img.size(); i++) {
		findContours(binary_img[i], contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		Mat contour_output = Mat::zeros(binary_img[i].size(), CV_8UC3);
		cout << "The number of contours detected is: " << contours.size() << endl;
		opencv_contours.push_back(contours[0]);
		if (contours.size() > 0) {
			for (int i = 0; i < contours.size(); i++)
			{
				double area = contourArea(contours[i]);
				if (area > maxsize) {
					maxsize = area;
					maxind = i;
					boundrec = boundingRect(contours[i]);
				}
			}
			//drawContours(contour_output, contours, -1, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
			drawContours(contour_output, contours, -1, Scalar(0, 255, 0), 2, 8, hierarchy);

			contour_img.push_back(contour_output);
		}
	}
	Mat res;

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(4);

	for (int i = 0; i < contour_img.size(); i++) {
		string num = std::to_string(i);
		string s = "opencv_contour" + num;
		//imshow(s, contour_img[i]);
		imwrite("image_" + filename + "p2_opencv" + num + ".png", contour_img[i]);

	}
	for (int i = 0; i < binary_img.size(); i++) {
		string num = std::to_string(i);
		string s = "binary_image" + num;
		//imshow(s, binary_img[i]);
	}
	return opencv_contours;
}

void p2_evaluate(vector<vector<Point>> opencv_contours, vector<vector<Point>> diy_contours) {
	double tp = 0;
	double fp = 0;
	for (int i = 0; i < diy_contours.size(); i++) {
		for (int j = 0; j < diy_contours[i].size(); j++) {
			Point val = diy_contours[i][j];
			vector<Point>::iterator result = find(opencv_contours[i].begin(), opencv_contours[i].end(), val);
			if (result == opencv_contours[i].end())
				fp++;
			else
				tp++;
		}
	}
	double res = tp / (tp + fp);
	cout << "tp is : " << tp << " fp is : " << fp << endl;
	cout << "The accuracy is " << res << endl;
}
Mat p1(Mat img) {
	int rows = img.rows;
	int cols = img.cols;
	int num = 0;
	map<vector<int>, int> colorMap;
	//scan 1st
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Vec3b color = img.at<Vec3b>(Point(i, j));
			vector<int> colorvec = { color[0] / 10 * 10, color[1] / 10 * 10, color[2] / 10 * 10 };
			if (colorMap.find(colorvec) == colorMap.end())
				colorMap[colorvec] = 1;
			else
				colorMap[colorvec]++;
		}
	}
	// colorlist
	priority_queue<pair<int, vector<int>>>  colorQue;
	for (auto c : colorMap) {
		colorQue.push(make_pair(c.second, c.first));
	}
	int total_value = rows * cols;
	vector<vector<int>> colorList;
	map<vector<int>, int> temp;
	while (!colorQue.empty()) {
		auto val = colorQue.top();
		colorQue.pop();
		if (val.first < total_value * 0.01)
			continue;
		temp[val.second] = val.first;
		colorList.push_back(val.second);
	}
	colorMap = temp;
	for (int i = 0; i < colorList.size(); i++) {
		for (int j = i + 1; j < colorList.size(); j++) {
			int b = abs(colorList[i][0] - colorList[j][0]);
			int g = abs(colorList[i][1] - colorList[j][1]);
			int r = abs(colorList[i][2] - colorList[j][2]);
			if (b < 15 && g < 15 && r < 15) {
				colorList.erase(colorList.begin() + j);
				j--;
			}
		}
	}
	map<vector<int>, int> temp1;
	for (int i = 0; i < colorList.size(); i++) {
		if (temp.find(colorList[i]) != temp.end())  //find
			temp1[colorList[i]] = temp[colorList[i]];
	}
	colorMap = temp1;
	//scan 2nd
	Mat res_img(rows, cols, CV_8UC1);
	vector<Point>blob;
	int count = 0;
	int max_area_number = 0;
	vector<int> bgcolor;
	for (auto c : colorMap) {
		if (c.second >= max_area_number) {
			max_area_number = c.second;
			bgcolor = c.first;
		}
	}
	for (auto c : colorMap) {
		vector<Point> b;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				Vec3b color = img.at<Vec3b>(Point(i, j));
				vector<int> colorv = { color[0] / 10 * 10, color[1] / 10 * 10, color[2] / 10 * 10 };
				int bdiff = abs(colorv[0] - c.first[0]);
				int gdiff = abs(colorv[1] - c.first[1]);
				int rdiff = abs(colorv[2] - c.first[2]);
				if (bdiff <= 15 && gdiff <= 15 && rdiff <= 15) {
					// put the pixel to the current blob
					res_img.at<uchar>(i, j) = count;
					if (c.first != bgcolor)    //if the color is the background color, don't put it into the blob
						blob.push_back(Point(i, j));
				}
			}
		}
		if (blob.size() > 0)
			Blob.push_back(blob);
		blob.swap(b);
		count++;
	}
	cout << "p1pre count is : " << count << endl;
	return res_img;
}


int start(string filename) {
	vector<vector<Point>>blank;
	vector<vector<Point>>blank1;
	vector<vector<Point>>blank2;
	Blob.swap(blank);
	att_borders.swap(blank1);
	rest_borders.swap(blank2);
	shape_Type.clear();
	attach_Type.clear();

	Mat img = imread(filename, IMREAD_COLOR);
	Mat gray_img = img.clone();
	Mat p1_img = img.clone();
	Mat p2_opencv_img = img.clone();
	Mat p2_diy_img = img.clone();
	Mat res_img;

	vector<vector<Point>> opencv_contours;
	vector<vector<Point>> diy_contours;

	if (!img.data) {
		cout << "File not found" << std::endl;
		return -1;
	}

	res_img = p1(p1_img);

	namedWindow("Original");
	//imshow("Original", img);

	opencv_contours = p2_opencv(p2_opencv_img, res_img);
	diy_contours = p2_diy(p2_diy_img, res_img);
	cout << "opencv_contours: " << opencv_contours.size() << endl;
	cout << "diy_contours: " << diy_contours.size() << endl;
	p2_evaluate(opencv_contours, diy_contours);
	p3_objectAgainst(img, res_img, opencv_contours);
	attach_print(img, diy_contours);
	p5_shapeClassification(diy_contours, res_img);

}
int main(int argc, char **argv) {
	for (int i = 1027; i < 1028; i++) {
		filename =  to_string(i) + ".png";
		start(filename);
	}

	waitKey(0);

	return 0;
}

