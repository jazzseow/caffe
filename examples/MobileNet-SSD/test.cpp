#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void display(Mat img){
	for (int h = 0; h < img.rows; h++){
		for (int w = 0; w < img.cols; w++){
			cout<<(int)img.at<Vec3b>(h,w)[0]<<", "<<
			(int)img.at<Vec3b>(h,w)[1]<<", "<<
			(int)img.at<Vec3b>(h,w)[2]<<endl;
		}
	}
	cout<<endl;
}
int main(){
		Mat img_mat = imread("/home/jazz/SSD/caffe/examples/MobileNet-SSD/images/000001.jpg");

	Mat cpy;

	resize(img_mat, cpy, Size(3,5));
	display(cpy);

	transpose(cpy,cpy);
	display(cpy);

	namedWindow( "Display window", WINDOW_NORMAL);// Create a window for display.
  imshow( "Display window", cpy);                   // Show our image inside it.

  waitKey(0);
	return 0;
}
