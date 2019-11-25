#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

string model = "G:/project/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
string config = "G:/project/ssd_mobilenet_v2_coco_2018_03_29/graph.pbtxt";
void dnn_tf();
//void dnn_ir_tf();
int main(int argc, char** argv) {
	dnn_tf();
	//dnn_ir_tf();
	cv::waitKey(0);
	destroyAllWindows();
	return 0;
}

void dnn_tf() {
	Mat src = imread("G:/images/pictures/dog.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return;
	}
	imshow("input", src);

	Net net = readNetFromTensorflow(model, config);
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(DNN_TARGET_CPU);

	printf("ssd network model loaded...\n");
	Mat blob = blobFromImage(src, 1.0, Size(300, 300), Scalar(), true, false, 5);
	net.setInput(blob);

	float threshold = 0.5;
	Mat detection = net.forward();

	// 获取推断时间
	vector<double> layerTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layerTimings) / freq;
	ostringstream ss;
	ss << "infernece : " << time << " ms";
	putText(src, ss.str(), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8);

	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > threshold) {
			size_t obj_index = (size_t)detectionMat.at<float>(i, 1);
			float tl_x = detectionMat.at<float>(i, 3) *  src.cols;
			float tl_y = detectionMat.at<float>(i, 4) *  src.rows;
			float br_x = detectionMat.at<float>(i, 5) *  src.cols;
			float br_y = detectionMat.at<float>(i, 6) *  src.rows;
			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int(br_y - tl_y)));
			rectangle(src, object_box, Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	imshow("ssd detection", src);
}

//void dnn_ir_tf() {
//	string xml = "G:/projects/models/frozen_inference_graph.xml";
//	string bin = "G:/projects/models/frozen_inference_graph.bin";
//	Mat src = imread("G:/images/pictures/dog.jpg");
//	if (src.empty()) {
//		printf("could not load image...\n");
//		return;
//	}
//	imshow("input", src);
//
//	Net net = readNetFromModelOptimizer(xml, bin);
//	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
//	net.setPreferableTarget(DNN_TARGET_CPU);
//
//	printf("ssd network model loaded...\n");
//	Mat blob = blobFromImage(src, 1.0, Size(300, 300), Scalar(), true, false, 5);
//	net.setInput(blob);
//
//	float threshold = 0.5;
//	Mat detection = net.forward();
//
//	// 获取推断时间
//	vector<double> layerTimings;
//	double freq = getTickFrequency() / 1000;
//	double time = net.getPerfProfile(layerTimings) / freq;
//	ostringstream ss;
//	ss << "infernece : " << time << " ms";
//	putText(src, ss.str(), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8);
//
//	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
//	for (int i = 0; i < detectionMat.rows; i++) {
//		float confidence = detectionMat.at<float>(i, 2);
//		if (confidence > threshold) {
//			size_t obj_index = (size_t)detectionMat.at<float>(i, 1);
//			float tl_x = detectionMat.at<float>(i, 3) *  src.cols;
//			float tl_y = detectionMat.at<float>(i, 4) *  src.rows;
//			float br_x = detectionMat.at<float>(i, 5) *  src.cols;
//			float br_y = detectionMat.at<float>(i, 6) *  src.rows;
//			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int(br_y - tl_y)));
//			rectangle(src, object_box, Scalar(0, 0, 255), 2, 8, 0);
//		}
//	}
//	imshow("ssd model optimizer", src);
//}