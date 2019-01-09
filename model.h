#pragma once
#include <dlib/dnn.h>
#include <dlib/image_processing/full_object_detection.h>

using namespace dlib;
using namespace std;

struct face_dectection {
    rectangle rect;
    matrix<rgb_pixel> face;
    matrix<float,0,1> descriptor;
    full_object_detection shape;
};

class model {
    const string basedir;
    const int face_landmarks;
    const bool use_mmod;
    const long pyramid_upsize;
    const int yolo_type;
public:
    model(const string& basedir, int landmarks, bool use_mmod, long upsize, int yolo_type);

    std::vector<rectangle> detect_faces(matrix<rgb_pixel>& img);
    int predict_objects(matrix<rgb_pixel>& img, float thresh, float nms);
    std::vector<face_dectection> predict_faces(matrix<rgb_pixel>& img, int mark_thickness, rgb_pixel mark_color = rgb_pixel(255, 0, 0));
    std::vector<face_dectection> filter_faces(const std::vector<face_dectection>& dets, const matrix<float,0,1>& selected, float thres, bool match_all);
};
