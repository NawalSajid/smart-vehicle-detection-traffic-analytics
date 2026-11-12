
#ifndef CENTROID_TRACKER_H
#define CENTROID_TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

using namespace std;
using namespace cv;

class CentroidTracker
{
private:

    int nextObjectID;

    int maxDisappeared;
    double maxDistance;

    map<int, Point> objects;
    map<int, int> disappeared;

public:

    CentroidTracker(int maxDisappeared = 10,
                    double maxDistance = 80.0);

    void registerObject(Point centroid);

    void deregisterObject(int objectID);

    map<int, Point> update(const vector<Rect>& detections);
};

#endif
