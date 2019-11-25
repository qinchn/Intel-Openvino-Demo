#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace InferenceEngine;
using namespace std;

int main(int argc, char** argv) {
	string xml = "G:/project/models/vehicle-license-plate-detection-barrier-0106/FP32/pedestrian-detection-adas-0002.xml";
	string bin = "G:/project/models/vehicle-license-plate-detection-barrier-0106/FP32/pedestrian-detection-adas-0002.bin";
	Mat src = imread("G:/images/pictures/crowd.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("huamn-detection", WINDOW_AUTOSIZE);
	imshow("input", src);

	int image_width = src.cols;
	int image_height = src.rows;

	vector<file_name_t> dirs;
	std::string s("C:\\Intel\\openvino_2019.1.148\\deployment_tools\\inference_engine\\bin\\intel64\\Debug");
	std::wstring ws;
	ws.assign(s.begin(), s.end());
	dirs.push_back(ws);

	// 创建IE插件
	InferenceEnginePluginPtr engine_ptr = PluginDispatcher(dirs).getSuitablePlugin(TargetDevice::eCPU);
	InferencePlugin plugin(engine_ptr);

	// 加载CPU扩展支持
	plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

	// 加载网络
	CNNNetReader network_reader;
	network_reader.ReadNetwork(xml);
	network_reader.ReadWeights(bin);

	// 获取输入输出
	auto network = network_reader.getNetwork();
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

	// 设置输入输出
	for (auto &item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);
	}

	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}

	// 创建可执行网络
	auto exec_network = plugin.LoadNetwork(network, {});

	// 请求推断
	auto infer_request = exec_network.CreateInferRequest();

	// 设置输入数据
	for (auto &item : input_info) {
		auto inut_name = item.first;
		auto input = infer_request.GetBlob(inut_name);
		size_t num_channels = input->getTensorDesc().getDims()[1];
		size_t h = input->getTensorDesc().getDims()[2];
		size_t w = input->getTensorDesc().getDims()[3];
		size_t image_size = w * h;
		Mat blob_image;
		resize(src, blob_image, Size(w, h));

		// NCHW
		unsigned char* data = static_cast<unsigned char*>(input->buffer());
		for (size_t row = 0; row < h; row++) {
			for (size_t col = 0; col < w; col++) {
				for (size_t ch = 0; ch < num_channels; ch++) {
					data[image_size*ch + row * w + col] = blob_image.at<Vec3b>(row, col)[ch];
				}
			}
		}
	}

	// 执行预测
	infer_request.Infer();

	// 处理预测结果 [1x1xNx7]
	for (auto &item : output_info) {
		auto output_name = item.first;
		auto output = infer_request.GetBlob(output_name);
		const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
		const SizeVector outputDims = output->getTensorDesc().getDims();
		const int rows = outputDims[2];
		const int object_size = outputDims[3];
		for (int row = 0; row < rows; row++) {
			float label = detection[row*object_size + 1];
			float confidence = detection[row*object_size + 2];
			float x_min = detection[row*object_size + 3] * image_width;
			float y_min = detection[row*object_size + 4] * image_height;
			float x_max = detection[row*object_size + 5] * image_width;
			float y_max = detection[row*object_size + 6] * image_height;
			if (confidence > 0.5) {
				Rect object_box((int)x_min, (int)y_min, (int)(x_max - x_min), (int(y_max - y_min)));
				rectangle(src, object_box, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}
	imshow("huamn-detection", src);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
