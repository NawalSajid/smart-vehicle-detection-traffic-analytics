
#include "CentroidTracker.h"
#include <cmath>
#include <limits>

CentroidTracker::CentroidTracker(int maxDisappeared,
                                 double maxDistance)
{
    nextObjectID = 0;
    this->maxDisappeared = maxDisappeared;
    this->maxDistance = maxDistance;
}

void CentroidTracker::registerObject(Point centroid)
{
    objects[nextObjectID] = centroid;
    disappeared[nextObjectID] = 0;
    nextObjectID++;
}

void CentroidTracker::deregisterObject(int objectID)
{
    objects.erase(objectID);
    disappeared.erase(objectID);
}

map<int, Point> CentroidTracker::update(
    const vector<Rect>& detections)
{
    vector<Point> centroids;

    for (const Rect& box : detections)
    {
        centroids.push_back(Point(
            box.x + box.width / 2,
            box.y + box.height / 2
        ));
    }

    // No existing tracks: register all detections.
    if (objects.empty())
    {
        for (const Point& centroid : centroids)
            registerObject(centroid);

        return objects;
    }

    // No detections: age existing tracks.
    if (centroids.empty())
    {
        vector<int> expired;

        for (auto& object : objects)
        {
            disappeared[object.first]++;

            if (disappeared[object.first] > maxDisappeared)
                expired.push_back(object.first);
        }

        for (int id : expired)
            deregisterObject(id);

        return objects;
    }

    vector<int> objectIDs;
    vector<bool> objectMatched;
    vector<bool> detectionMatched;

    for (const auto& object : objects)
    {
        objectIDs.push_back(object.first);
        objectMatched.push_back(false);
    }

    detectionMatched.assign(centroids.size(), false);

    // Match each existing vehicle to its nearest detection.
    while (true)
    {
        double smallestDistance = numeric_limits<double>::max();
        int bestObject = -1;
        int bestDetection = -1;

        for (int i = 0; i < (int)objectIDs.size(); i++)
        {
            if (objectMatched[i])
                continue;

            Point oldPoint = objects[objectIDs[i]];

            for (int j = 0; j < (int)centroids.size(); j++)
            {
                if (detectionMatched[j])
                    continue;

                double distance = norm(oldPoint - centroids[j]);

                if (distance < smallestDistance)
                {
                    smallestDistance = distance;
                    bestObject = i;
                    bestDetection = j;
                }
            }
        }

        if (bestObject == -1 ||
            smallestDistance > maxDistance)
            break;

        int id = objectIDs[bestObject];

        objects[id] = centroids[bestDetection];
        disappeared[id] = 0;

        objectMatched[bestObject] = true;
        detectionMatched[bestDetection] = true;
    }

    // Age unmatched existing vehicles.
    vector<int> expired;

    for (int i = 0; i < (int)objectIDs.size(); i++)
    {
        if (!objectMatched[i])
        {
            int id = objectIDs[i];
            disappeared[id]++;

            if (disappeared[id] > maxDisappeared)
                expired.push_back(id);
        }
    }

    for (int id : expired)
        deregisterObject(id);

    // Register unmatched new detections.
    for (int j = 0; j < (int)centroids.size(); j++)
    {
        if (!detectionMatched[j])
            registerObject(centroids[j]);
    }

    return objects;
}
