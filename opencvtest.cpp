#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;


// Function to load class names from coco.names file
vector<string> getClassNames(const string& classesFile) {
    vector<string> classNames;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

// Function to draw detected objects
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, const vector<string>& classNames) {
    // Draw a bounding box.
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);

    // Get the label for the class name and its confidence.
    string label = format("%.2f", conf);
    if (!classNames.empty()) {
        label = classNames[classId] + ": " + label;
    }

    // Display the label at the top of the bounding box.
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
}

int main() {

    // path to yolo files
    // change based on your pathing
    string modelConfiguration = "C:/Users/User/source/repos/opencvtest/yolo/yolov4.cfg";
    string modelWeights = "C:/Users/User/source/repos/opencvtest/yolo/yolov4.weights";
    string classesFile = "C:/Users/User/source/repos/opencvtest/yolo/coco.names";

    // Load class names
    vector<string> classNames = getClassNames(classesFile);

    // Load YOLO model
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    // TODO : You can change this to DNN_TARGET_CUDA if you have a GPU
    net.setPreferableTarget(DNN_TARGET_CPU);

    // loading youtube stream here
    // check youtube links file
    // youtube links 3 live
    //std::string videoURL = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1728068780/ei/TOj_ZrrpHLiO6dsPk7-KuQI/ip/92.40.196.210/id/rnXIjl_Rzy4.1/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hdlc/1/hls_chunk_host/rr1---sn-uigxx03-2hxe.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/spc/54MbxX4QnEQGTm3lw_t0HKt1e-30KijBg-g4-0VytAikdd4qCtzl/vprv/1/playlist_type/DVR/initcwndbps/736250/mh/K2/mm/44/mn/sn-uigxx03-2hxe/ms/lva/mv/m/mvi/1/pl/23/dover/11/pacing/0/keepalive/yes/fexp/51300761/mt/1728046839/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,hdlc,xpc,spc,vprv,playlist_type/sig/AJfQdSswRQIgByGofHV38iRheKeZiQH22-I8DTzwhRN1kqQKNMMebZYCIQD641WW4Upc9u0vCbHJuzPRISUQ1CVYFKdcQNfmValT0w%3D%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/ABPmVW0wRQIhALc1d4AeJ1Nxn4NPrjbd33XGfTVh5qbeOa5ZUXeFLzatAiA1HdmgYTY-cxoeaT2EDHANjRj4BsjamuImj2CwrYE4HQ%3D%3D/playlist/index.m3u8";
    //cv::VideoCapture cap(videoURL);

    // local camera access
    cv::VideoCapture cap(0);

    // Check if the video stream opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream from URL" << std::endl;
        return -1;
    }

    cv::Mat frame, prevFrame, diffFrame, grayFrame;

    // Process each frame
    while (true) {
        // Capture each frame
        bool ret = cap.read(frame);
        if (!ret) {
            std::cerr << "Error: Cannot read frame from stream" << std::endl;
            break;
        }

        // Preprocess the frame for YOLO (resize, normalization, etc.)
        Mat blob;
        blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // Run forward pass on the YOLO network
        vector<Mat> outs;
        vector<string> outNames = net.getUnconnectedOutLayersNames();
        net.forward(outs, outNames);

        // Post-process YOLO output to extract bounding boxes and labels
        float confThreshold = 0.5;  // Confidence threshold
        float nmsThreshold = 0.4;   // Non-max suppression threshold

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        // Perform Non-Max Suppression to eliminate redundant boxes
        vector<int> indices;
        dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        // Draw bounding boxes for detected objects
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classNames);
        }

        // Display the video and the motion detection result
        cv::imshow("YOLO Object Detection", frame);

        // Break the loop when 'q' key is pressed
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

