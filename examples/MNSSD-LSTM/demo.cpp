#include <caffe/caffe.hpp>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace caffe;
using namespace std;
using namespace cv;

float CONF_THRESH = 0.2;
class Detector {
	public:
		Detector(const string& model_file, const string& weights_file);
		void Detect(string im_name);


	private:
		boost::shared_ptr<Net<float> > net_;
		void vis_detections(cv::Mat image, vector<vector<float> > detections, float confidence_threshold);
		Detector(){}
};

Detector::Detector(const string& model_file, const string& weights_file)
{
	net_ = boost::shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(weights_file);
}

void Detector::Detect(string im_name){
	Mat origImg = imread(im_name);
	Mat resizeImg;

	LOG(INFO);

	resize(origImg, resizeImg, Size(300, 300));
	LOG(INFO);
	Mat copyImg(300, 300, CV_32FC3, cv::Scalar(0, 0, 0));
	LOG(INFO);
	// Mat copyImg;
	// resize(origImg, copyImg, Size(300,300));

	//BGR
	for (int h = 0; h < 300; h++){
		for (int w = 0; w < 300; w++){
				copyImg.at<Vec3f>(Point(w, h))[0] = (float(resizeImg.at<Vec3b>(Point(w, h))[0]) - float(127.5)) * float(0.007843);
				copyImg.at<Vec3f>(Point(w, h))[1] =	(float(resizeImg.at<Vec3b>(Point(w, h))[1]) - float(127.5)) * float(0.007843);
				copyImg.at<Vec3f>(Point(w, h))[2] = (float(resizeImg.at<Vec3b>(Point(w, h))[2]) - float(127.5)) * float(0.007843);
			}
	}
	// for (int h = 0; h < origImg.rows; h++){
	// 	for (int w = 0; w < origImg.cols; w++){
	// 			cout << (float)copyImg.at<Vec3f>(h,w)[0]<<", "<<
	// 		(float)copyImg.at<Vec3f>(h,w)[1]<<", "<<
	// 		(float)copyImg.at<Vec3f>(h,w)[2]<<endl;
	// 	}
	// }
	// LOG(INFO);
	
	cv::FileStorage fs("/home/jazz/SSD/caffe/demo_cpp.xml", cv::FileStorage::WRITE);
	fs << "copyImg" << copyImg;
	fs.release();
	waitKey(0);
	
	LOG(INFO);
	int height = 300;
	int width = 300;
	float *data_buf = new float[300 * 300 * 3];

	for (int h = 0; h < height; h++){
		for (int w = 0; w < width; w++){
			data_buf[(0 * height + h)*width + w] = float(copyImg.at<Vec3f>(Point(w, h))[0]);
			data_buf[(1 * height + h)*width + w] = float(copyImg.at<Vec3f>(Point(w, h))[1]);
			data_buf[(2 * height + h)*width + w] = float(copyImg.at<Vec3f>(Point(w, h))[2]);
		}
	}
	LOG(INFO);
	net_->blob_by_name("data")->set_cpu_data(data_buf);

	net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result + 1, result + 7);
    detections.push_back(detection);
    result += 7;
  }
	LOG(INFO);
  vis_detections(origImg, detections, CONF_THRESH);
	LOG(INFO);
}

void Detector::vis_detections(cv::Mat image, vector<vector<float> > detections, float confidence_threshold)
{

  const string CLASSES[21] = {"__background__",
             "aeroplane", "bicycle", "bird", "boat",
             "bottle", "bus", "car", "cat", "chair",
             "cow", "diningtable", "dog", "horse",
             "motorbike", "person", "pottedplant",
             "sheep", "sofa", "train", "tvmonitor"};

  for (int i = 0; i < detections.size(); ++i) {
    const vector<float>& d = detections[i];
    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
    CHECK_EQ(d.size(), 6);
    const float score = d[1];
    if (score >= confidence_threshold) {
      std::ostringstream strs;
      strs << CLASSES[static_cast<int>(d[0])]<< ": " << score;
      std::string str = strs.str();
			LOG(INFO)<<str<<endl;
      cv::rectangle(image, cv::Point(static_cast<int>(d[2] * image.cols), static_cast<int>(d[3] * image.rows)), cv::Point(static_cast<int>(d[4] * image.cols), static_cast<int>(d[5] * image.rows)), cv::Scalar(255, 0, 0));
			cv::putText(image, str, cv::Point(static_cast<int>(d[2] * image.cols), static_cast<int>(d[3] * image.rows)),  cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255,255));
    }
  }

	namedWindow( "Display window", WINDOW_NORMAL);// Create a window for display.
  imshow( "Display window", image);                   // Show our image inside it.

  waitKey(0);
}

int main()
{
	string model_file = "examples/MobileNet-SSD/model_quantized/MobileNetSSD_quantized_deploy.prototxt";
	string weights_file = "examples/MobileNet-SSD/model_quantized/MobileNetSSD_quantized_deploy.caffemodel";

	// Caffe::set_mode(Caffe::GPU);
	Caffe::set_mode(Caffe::CPU);
	Detector det = Detector(model_file, weights_file);
	string im_names="examples/MobileNet-SSD/images/horse.jpg";
	// clock_t t1 = clock();
	det.Detect(im_names);
	// // det.Detect_video("examples/images/test.avi");
	return 0;
}
