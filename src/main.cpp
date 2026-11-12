
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "CentroidTracker.h"

int main()
{
    // 1. Open traffic video

    std::string videoPath = "videos/traffic.mp4";

    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open traffic video!" << std::endl;
        return -1;
    }

    int frameWidth = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Video opened successfully!" << std::endl;
    std::cout << "Width: " << frameWidth << std::endl;
    std::cout << "Height: " << frameHeight << std::endl;
    std::cout << "FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;

    // 2. Load YOLOv8n model

    std::cout << "\nLoading YOLOv8n model..." << std::endl;

    cv::dnn::Net net =
        cv::dnn::readNetFromONNX("models/yolov8n.onnx");

    if (net.empty())
    {
        std::cerr << "Error: Could not load YOLO model!" << std::endl;
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::cout << "YOLOv8n model loaded successfully!" << std::endl;

    // 3. Vehicle classes

    std::vector<int> vehicleClasses = {2, 3, 5, 7};

    auto getClassName = [](int classId) -> std::string
    {
        switch (classId)
        {
            case 2: return "Car";
            case 3: return "Motorcycle";
            case 5: return "Bus";
            case 7: return "Truck";
            default: return "Unknown";
        }
    };

    // 4. Detection settings

    const int inputSize = 640;

    const float confidenceThreshold = 0.30f;
    const float nmsThreshold = 0.45f;

    // 5. Initialize tracker and vehicle counters

    CentroidTracker tracker(5, 150.0);

    int totalVehicles = 0;
    int carsCount = 0;
    int busesCount = 0;
    int trucksCount = 0;
    int motorcyclesCount = 0;

    // Horizontal counting line in original video coordinates
    int countingLineY = frameHeight / 2;

    // Remember previous vehicle positions
    std::map<int, int> previousY;

    // Prevent counting the same vehicle more than once
    std::set<int> countedIDs;

    // 6. Process video frames

    cv::Mat frame;

    while (cap.read(frame))
    {
        if (frame.empty())
            break;

        // 7. Resize frame while preserving aspect ratio

        float scale = std::min(
            (float)inputSize / frame.cols,
            (float)inputSize / frame.rows
        );

        int resizedWidth = (int)(frame.cols * scale);
        int resizedHeight = (int)(frame.rows * scale);

        cv::Mat resizedFrame;

        cv::resize(
            frame,
            resizedFrame,
            cv::Size(resizedWidth, resizedHeight)
        );

        int padLeft = (inputSize - resizedWidth) / 2;
        int padTop = (inputSize - resizedHeight) / 2;

        cv::Mat inputFrame;

        cv::copyMakeBorder(
            resizedFrame,
            inputFrame,
            padTop,
            inputSize - resizedHeight - padTop,
            padLeft,
            inputSize - resizedWidth - padLeft,
            cv::BORDER_CONSTANT,
            cv::Scalar(114, 114, 114)
        );

        // 8. Convert frame into YOLO input blob

        cv::Mat blob = cv::dnn::blobFromImage(
            inputFrame,
            1.0 / 255.0,
            cv::Size(inputSize, inputSize),
            cv::Scalar(),
            true,
            false
        );

        // 9. Run YOLO inference

        net.setInput(blob);

        cv::Mat output = net.forward();

        if (output.dims != 3 ||
            output.size[1] != 84 ||
            output.size[2] != 8400)
        {
            std::cerr << "Unexpected YOLO output dimensions!" << std::endl;
            return -1;
        }

        cv::Mat outputData = output.isContinuous()
            ? output
            : output.clone();

        cv::Mat output2D(
            outputData.size[1],
            outputData.size[2],
            CV_32F,
            outputData.ptr<float>()
        );

        cv::Mat predictions;

        cv::transpose(output2D, predictions);

        // 10. Collect vehicle detections

        std::vector<std::vector<cv::Rect>> boxesByClass(80);
        std::vector<std::vector<float>> scoresByClass(80);

        for (int i = 0; i < predictions.rows; i++)
        {
            float* data = predictions.ptr<float>(i);

            float confidence = 0.0f;
            int classId = -1;

            for (int j = 4; j < 84; j++)
            {
                if (data[j] > confidence)
                {
                    confidence = data[j];
                    classId = j - 4;
                }
            }

            if (confidence < confidenceThreshold)
                continue;

            if (std::find(
                vehicleClasses.begin(),
                vehicleClasses.end(),
                classId
            ) == vehicleClasses.end())
                continue;

            float centerX = data[0];
            float centerY = data[1];
            float width = data[2];
            float height = data[3];

            int left = (int)(
                (centerX - width / 2 - padLeft) / scale
            );

            int top = (int)(
                (centerY - height / 2 - padTop) / scale
            );

            int right = (int)(
                (centerX + width / 2 - padLeft) / scale
            );

            int bottom = (int)(
                (centerY + height / 2 - padTop) / scale
            );

            left = std::max(0, std::min(left, frame.cols));
            top = std::max(0, std::min(top, frame.rows));

            right = std::max(0, std::min(right, frame.cols));
            bottom = std::max(0, std::min(bottom, frame.rows));

            if (right <= left || bottom <= top)
                continue;

            boxesByClass[classId].push_back(
                cv::Rect(left, top, right - left, bottom - top)
            );

            scoresByClass[classId].push_back(confidence);
        }

        // 11. Apply Non-Maximum Suppression

        std::vector<cv::Rect> finalBoxes;
        std::vector<float> finalScores;
        std::vector<int> finalClassIds;

        for (int classId : vehicleClasses)
        {
            std::vector<int> indices;

            cv::dnn::NMSBoxes(
                boxesByClass[classId],
                scoresByClass[classId],
                confidenceThreshold,
                nmsThreshold,
                indices
            );

            for (int index : indices)
            {
                finalBoxes.push_back(boxesByClass[classId][index]);
                finalScores.push_back(scoresByClass[classId][index]);
                finalClassIds.push_back(classId);
            }
        }

        // 12. Update vehicle tracking

        std::map<int, cv::Point> trackedObjects =
            tracker.update(finalBoxes);

        // 13. Resize frame for display

        const int displayWidth = 1280;
        const int displayHeight = 720;

        float displayScale = std::min(
            (float)displayWidth / frame.cols,
            (float)displayHeight / frame.rows
        );

        cv::Mat displayFrame;

        cv::resize(
            frame,
            displayFrame,
            cv::Size(
                (int)(frame.cols * displayScale),
                (int)(frame.rows * displayScale)
            )
        );

        // 14. Draw vehicle detections

        int vehicleCount = 0;

        for (int i = 0; i < (int)finalBoxes.size(); i++)
        {
            cv::Rect box = finalBoxes[i];

            int x = (int)(box.x * displayScale);
            int y = (int)(box.y * displayScale);

            int w = (int)(box.width * displayScale);
            int h = (int)(box.height * displayScale);

            cv::rectangle(
                displayFrame,
                cv::Rect(x, y, w, h),
                cv::Scalar(0, 255, 0),
                2
            );

            std::string label =
                getClassName(finalClassIds[i]) + " " +
                std::to_string((int)(finalScores[i] * 100)) + "%";

            cv::putText(
                displayFrame,
                label,
                cv::Point(x, std::max(y - 8, 20)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 0),
                2
            );

            vehicleCount++;
        }

        // 15. Vehicle counting logic

        // Only process tracks that match a detection in this frame.
        // This prevents temporarily disappeared tracks from being counted.

        for (const auto& object : trackedObjects)
        {
            int objectID = object.first;
            cv::Point centroid = object.second;

            int bestIndex = -1;
            double bestDistance = 2.0;

            for (int i = 0; i < (int)finalBoxes.size(); i++)
            {
                cv::Point boxCenter(
                    finalBoxes[i].x + finalBoxes[i].width / 2,
                    finalBoxes[i].y + finalBoxes[i].height / 2
                );

                double distance = cv::norm(centroid - boxCenter);

                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestIndex = i;
                }
            }

            if (bestIndex == -1)
                continue;

            int currentY = centroid.y;

            // First observation of this vehicle
            if (previousY.find(objectID) == previousY.end())
            {
                previousY[objectID] = currentY;
                continue;
            }

            int oldY = previousY[objectID];

            // Check whether vehicle crossed the horizontal line
            bool crossedLine =
                (oldY < countingLineY && currentY >= countingLineY) ||
                (oldY > countingLineY && currentY <= countingLineY);

            if (crossedLine && countedIDs.find(objectID) == countedIDs.end())
            {
                countedIDs.insert(objectID);

                totalVehicles++;

                int classId = finalClassIds[bestIndex];

                switch (classId)
                {
                    case 2:
                        carsCount++;
                        break;

                    case 3:
                        motorcyclesCount++;
                        break;

                    case 5:
                        busesCount++;
                        break;

                    case 7:
                        trucksCount++;
                        break;
                }

                std::cout << "Vehicle ID " << objectID
                          << " crossed the counting line. Total: "
                          << totalVehicles << std::endl;
            }

            previousY[objectID] = currentY;
        }

        // 16. Draw tracking IDs

        for (const auto& object : trackedObjects)
        {
            int objectID = object.first;
            cv::Point centroid = object.second;

            int centerX = (int)(centroid.x * displayScale);
            int centerY = (int)(centroid.y * displayScale);

            cv::circle(
                displayFrame,
                cv::Point(centerX, centerY),
                4,
                cv::Scalar(0, 0, 255),
                -1
            );

            std::string idLabel = "ID: " + std::to_string(objectID);

            cv::putText(
                displayFrame,
                idLabel,
                cv::Point(centerX + 8, centerY),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 0, 255),
                2
            );
        }

        // 17. Draw virtual counting line

        int displayLineY = (int)(countingLineY * displayScale);

        cv::line(
            displayFrame,
            cv::Point(0, displayLineY),
            cv::Point(displayFrame.cols, displayLineY),
            cv::Scalar(255, 0, 255),
            2
        );

        cv::putText(
            displayFrame,
            "COUNTING LINE",
            cv::Point(20, displayLineY - 10),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(255, 0, 255),
            2
        );

        // 18. Display traffic statistics

        cv::putText(
            displayFrame,
            "Detected: " + std::to_string(vehicleCount),
            cv::Point(25, 35),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 255, 255),
            2
        );

        cv::putText(
            displayFrame,
            "Tracked: " + std::to_string(trackedObjects.size()),
            cv::Point(25, 70),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(255, 255, 0),
            2
        );

        cv::putText(
            displayFrame,
            "Total Counted: " + std::to_string(totalVehicles),
            cv::Point(25, 110),
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::putText(
            displayFrame,
            "Cars: " + std::to_string(carsCount) +
            "  Buses: " + std::to_string(busesCount),
            cv::Point(25, 150),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(255, 255, 255),
            2
        );

        cv::putText(
            displayFrame,
            "Trucks: " + std::to_string(trucksCount) +
            "  Motorcycles: " + std::to_string(motorcyclesCount),
            cv::Point(25, 185),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(255, 255, 255),
            2
        );

        // 19. Display video

        cv::imshow(
            "Smart Vehicle Detection and Traffic Analytics",
            displayFrame
        );

        if (cv::waitKey(1) == 27)
            break;
    }

    // 20. Release resources

    cap.release();
    cv::destroyAllWindows();

    std::cout << "\nVehicle detection, tracking and counting completed!"
              << std::endl;

    return 0;
}
