#include <inference_engine.hpp>
#include "ext_list.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

struct VehicleObject {
	std::string type;
	std::string color;
	cv::Rect location;
	int channelId;
};

struct LicensePlateObject {
	std::string text;
	cv::Rect location;
	int channelId;
};

struct MyDetectionNet {
	ExecutableNetwork net;
	std::string inputName;
	std::string sencondOutputName;
	std::string outputName;
};

template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
	InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
	const size_t width = blobSize[3];
	const size_t height = blobSize[2];
	const size_t channels = blobSize[1];
	T* blob_data = blob->buffer().as<T*>();

	cv::Mat resized_image(orig_image);
	if (width != orig_image.size().width || height != orig_image.size().height) {
		cv::resize(orig_image, resized_image, cv::Size(width, height));
	}

	int batchOffset = batchIndex * width * height * channels;

	for (size_t c = 0; c < channels; c++) {
		for (size_t h = 0; h < height; h++) {
			for (size_t w = 0; w < width; w++) {
				blob_data[batchOffset + c * width * height + h * width + w] =
					resized_image.at<cv::Vec3b>(h, w)[c];
			}
		}
	}
}

InferenceEngine::InferRequest vehicleplateDetectorinfer;
InferenceEngine::InferRequest vehicleAttrDetectorInfer;
InferenceEngine::InferRequest plateLicenseRecognizerInfer;

MyDetectionNet vehicleDetector;
MyDetectionNet attrDetector;
MyDetectionNet PLRDetector;

void loadVehiclePlateNetWork(InferencePlugin &plugin);
void loadVehicleAttributesNetWork(InferencePlugin &plugin);
void loadVehicleLicenseNetWork(InferencePlugin &plugin);

void fetchVehicleAttributes(Mat vehicle_roi, VehicleObject &info);
void fetchLicenseText(Mat plate_roi, LicensePlateObject &info);
const int padding = 5;
int main(int argc, char** argv) {
	Mat src = imread("G:/images/pictures/vehicle01.jpg");
	int image_height = src.rows;
	int image_width = src.cols;

	vector<file_name_t> dirs;
	std::string s("C:\\Intel\\openvino_2019.1.148\\deployment_tools\\inference_engine\\bin\\intel64\\Debug");
	// string to wstring
	std::wstring ws;
	ws.assign(s.begin(), s.end());
	dirs.push_back(ws);

	// 创建IE插件
	InferenceEnginePluginPtr engine_ptr = PluginDispatcher(dirs).getSuitablePlugin(TargetDevice::eCPU);
	InferencePlugin plugin(engine_ptr);

	// 加载CPU扩展库支持
	plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

	// 加载车辆与车牌检测网络
	loadVehiclePlateNetWork(plugin);

	// 加载车辆属性识别网络
	loadVehicleAttributesNetWork(plugin);

	// 加载车牌识别网络
	loadVehicleLicenseNetWork(plugin);

	/** Getting input blob **/
	auto input = vehicleplateDetectorinfer.GetBlob(vehicleDetector.inputName);
	size_t num_channels = input->getTensorDesc().getDims()[1];
	size_t h = input->getTensorDesc().getDims()[2];
	size_t w = input->getTensorDesc().getDims()[3];
	size_t image_size = h * w;
	Mat blob_image;
	resize(src, blob_image, Size(h, w));

	// NCHW
	unsigned char* data = static_cast<unsigned char*>(input->buffer());
	for (size_t row = 0; row < h; row++) {
		for (size_t col = 0; col < w; col++) {
			for (size_t ch = 0; ch < num_channels; ch++) {
				data[image_size*ch + row * w + col] = blob_image.at<Vec3b>(row, col)[ch];
			}
		}
	}

	// 执行预测
	vehicleplateDetectorinfer.Infer();

	// 获取输出数据
	auto output = vehicleplateDetectorinfer.GetBlob(vehicleDetector.outputName);
	const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
	const SizeVector outputDims = output->getTensorDesc().getDims();
	const int maxProposalCount = outputDims[2];
	const int objectSize = outputDims[3];

	// 解析输出结果
	for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
		float label = detection[curProposal * objectSize + 1];
		float confidence = detection[curProposal * objectSize + 2];
		float xmin = detection[curProposal * objectSize + 3] * image_width;
		float ymin = detection[curProposal * objectSize + 4] * image_height;
		float xmax = detection[curProposal * objectSize + 5] * image_width;
		float ymax = detection[curProposal * objectSize + 6] * image_height;
		if (confidence > 0.5) {
			printf("label id : %d\n", static_cast<int>(label));
			Rect rect;
			rect.x = static_cast<int>(xmin);
			rect.y = static_cast<int>(ymin);
			rect.width = static_cast<int>(xmax - xmin);
			rect.height = static_cast<int>(ymax - ymin);
			if (label == 2) { // 车牌
				Rect roi;
				roi.x = rect.x - padding;
				roi.y = rect.y - padding;
				roi.width = rect.width + padding * 2;
				roi.height = rect.height + padding * 2;
				LicensePlateObject lpo;
				lpo.location = roi;
				fetchLicenseText(src(roi), lpo);
				putText(src, lpo.text.c_str(), Point(roi.x - 40, roi.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2, 8);
			}
			if (label == 1) { // 车辆
				VehicleObject vo;
				vo.location = rect;
				fetchVehicleAttributes(src(rect), vo);
				putText(src, format("vehicle color: %s", vo.color.c_str()), Point(rect.x, rect.y - 20), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 255), 2, 8);
				putText(src, format("vehicle type: %s", vo.type.c_str()), Point(rect.x, rect.y - 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 255), 2, 8);
			}
			rectangle(src, rect, Scalar(0, 255, 255), 2, 8, 0);
		}
	}

	namedWindow("license_detection", WINDOW_NORMAL);
	imshow("license_detection", src);
	imwrite("G:/images/pictures/license01_detection.png", src);
	waitKey(0);
	destroyAllWindows();
	return 0;
}

void loadVehiclePlateNetWork(InferencePlugin &plugin) {
	// SSD 车辆检测模型
	CNNNetReader network_reader;
	network_reader.ReadNetwork("G:/project/models/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml");
	network_reader.ReadWeights("G:/project/models/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.bin");

	// 请求网络输入与输出信息
	auto network = network_reader.getNetwork();
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

	/**设置输入精度与维度**/
	for (auto &item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);
	}

	/** 设置输出精度与内容**/
	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}

	// 创建可执行网络对象
	auto executable_network = plugin.LoadNetwork(network, {});

	// 请求推断图
	vehicleplateDetectorinfer = executable_network.CreateInferRequest();
	vehicleDetector.net = executable_network;
	vehicleDetector.inputName = input_info.begin()->first;
	vehicleDetector.outputName = output_info.begin()->first;
}

void loadVehicleAttributesNetWork(InferencePlugin &plugin) {
	// SSD 车辆检测模型
	CNNNetReader network_reader;
	network_reader.ReadNetwork("G:/project/models/vehicle-license-plate-detection-barrier-0106/32/vehicle-attributes-recognition-barrier-0039.xml");
	network_reader.ReadWeights("G:/project/models/vehicle-license-plate-detection-barrier-0106/32/vehicle-attributes-recognition-barrier-0039.bin");

	// 请求网络输入与输出信息
	auto network = network_reader.getNetwork();
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

	// 72x72图像大小
	for (auto &item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);
	}

	// 输出精度
	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}

	// 创建可执行网络对象
	auto executable_network = plugin.LoadNetwork(network, {});

	// 请求推断图
	vehicleAttrDetectorInfer = executable_network.CreateInferRequest();
	attrDetector.net = executable_network;
	attrDetector.inputName = input_info.begin()->first;
	auto it = output_info.begin();
	attrDetector.outputName = (it++)->second->name;  // color is the first output
	attrDetector.sencondOutputName = (it++)->second->name;  // type is sencod output
}

void loadVehicleLicenseNetWork(InferencePlugin &plugin) {
	// 加载车牌识别网络
	CNNNetReader network_reader;
	network_reader.ReadNetwork("G:/project/models/vehicle-license-plate-detection-barrier-0106/16/license-plate-recognition-barrier-0001.xml");
	network_reader.ReadWeights("G:/project/models/vehicle-license-plate-detection-barrier-0106/16/license-plate-recognition-barrier-0001.bin");

	// 请求网络输入与输出信息
	auto network = network_reader.getNetwork();
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

	// 24x94, HxW图像大小
	InputInfo::Ptr& inputInfoFirst = input_info.begin()->second;
	inputInfoFirst->setInputPrecision(Precision::U8);
	inputInfoFirst->getInputData()->setLayout(Layout::NCHW);

	// 输出精度
	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}

	// 创建可执行网络对象
	auto executable_network = plugin.LoadNetwork(network, {});

	// 请求推断图
	plateLicenseRecognizerInfer = executable_network.CreateInferRequest();
	PLRDetector.net = executable_network;
	PLRDetector.inputName = input_info.begin()->first;
	auto sequenceInput = (++input_info.begin());
	PLRDetector.sencondOutputName = sequenceInput->first; // second input name
	PLRDetector.outputName = output_info.begin()->first;
}

void fetchVehicleAttributes(Mat vehicle_roi, VehicleObject &info) {
	Mat blob;
	resize(vehicle_roi, blob, Size(72, 72));
	static const std::string colors[] = {
		"white", "gray", "yellow", "red", "green", "blue", "black"
	};
	static const std::string types[] = {
		"car", "van", "truck", "bus"
	};
	// 设置输入数据
	Blob::Ptr inputBlob = vehicleAttrDetectorInfer.GetBlob(attrDetector.inputName);
	matU8ToBlob<uint8_t>(blob, inputBlob);

	// 执行推断图
	vehicleAttrDetectorInfer.Infer();

	// 获取输出结果
	auto colorsValues = vehicleAttrDetectorInfer.GetBlob(attrDetector.outputName)->buffer().as<float*>();
	auto typesValues = vehicleAttrDetectorInfer.GetBlob(attrDetector.sencondOutputName)->buffer().as<float*>();

	// 获取最大可能行
	const auto color_id = max_element(colorsValues, colorsValues + 7) - colorsValues;
	const auto type_id = max_element(typesValues, typesValues + 4) - typesValues;

	info.color = colors[color_id];
	info.type = types[type_id];
}

void fetchLicenseText(Mat plate_roi, LicensePlateObject &info) {
	static std::vector<std::string> items = {
		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
		"<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
		"<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
		"<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
		"<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
		"<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
		"<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
		"<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
		"<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
		"<Zhejiang>", "<police>",
		"A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
		"K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
		"U", "V", "W", "X", "Y", "Z"
	};
	// 设置输入数据
	Mat blob;
	resize(plate_roi, blob, Size(94, 24));
	Blob::Ptr inputBlob = plateLicenseRecognizerInfer.GetBlob(PLRDetector.inputName);
	matU8ToBlob<uint8_t>(blob, inputBlob);

	// 输入序列信息
	Blob::Ptr seqBlob = plateLicenseRecognizerInfer.GetBlob(PLRDetector.sencondOutputName);
	int maxSequenceSizePerPlate = seqBlob->getTensorDesc().getDims()[0];
	float* blob_data = seqBlob->buffer().as<float*>();
	blob_data[0] = 0.0f;
	std::fill(blob_data + 1, blob_data + maxSequenceSizePerPlate, 1.0f);

	// 执行推断
	plateLicenseRecognizerInfer.Infer();

	// 解析输出结果
	const auto data = plateLicenseRecognizerInfer.GetBlob(PLRDetector.outputName)->buffer().as<float*>();
	string result;
	for (int i = 0; i < maxSequenceSizePerPlate; i++) {
		if (data[i] == -1)
			break;
		result += items[data[i]];
	}
	info.text = result;
}