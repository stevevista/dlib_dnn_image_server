#include "yolo.h"
#include <cmath>

namespace darknet {

static float box_iou(box a, box b);

static bool nms_comparator(const detection& a, const detection& b)
{
    if(b.sort_class >= 0) {
        return a.prob[b.sort_class] > b.prob[b.sort_class];
    } else {
        return a.objectness > b.objectness;
    }
}

void do_nms_obj(std::vector<detection>& dets, float thresh)
{
    int total = dets.size();
    const int classes = total ? dets[0].prob.size() : 0;

    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i) {
        if(dets[i].objectness == 0) {
            std::swap(dets[i], dets[k]);
            --k;
            --i;
        }
    }
    total = k+1;

    for(i = 0; i < total; ++i) {
        dets[i].sort_class = -1;
    }

    std::sort(dets.begin(), dets.begin() + total, nms_comparator);
    for(i = 0; i < total; ++i){
        if(dets[i].objectness == 0) continue;
        box a = dets[i].bbox;
        for(j = i+1; j < total; ++j){
            if(dets[j].objectness == 0) continue;
            box b = dets[j].bbox;
            if (box_iou(a, b) > thresh){
                dets[j].objectness = 0;
                for(k = 0; k < classes; ++k){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}


void do_nms_sort(std::vector<detection>& dets, float thresh)
{
    int total = dets.size();
    const int classes = total ? dets[0].prob.size() : 0;

    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            std::swap(dets[i], dets[k]);
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }

        std::sort(dets.begin(), dets.begin() + total, nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

static float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

}
