#include "layers.h"

namespace darknet {

YoloLayer::YoloLayer(LayerPtr _linkin, const std::vector<float>& _biases) 
: linkin(_linkin)
, num(_biases.size()/2)
, biases(_biases)
{
    const int k = linkin->get_output().k();
    const int w = linkin->get_output().nc();
    const int h = linkin->get_output().nr();
    // calculate classes
    classes = k/num - 4 - 1;

    xy = alias_tensor(2*w*h);
    probs = alias_tensor((classes+1)*w*h);

    fprintf(stderr, "yolo(classes: %d)\n", classes);
}

const tensor& YoloLayer::forward_layer(const tensor& input)
{
    float *X = const_cast<float*>(input.host());
    tensor& out = const_cast<tensor&>(input);

    const int spatical = input.nr() * input.nc();

    for(int n = 0; n < num; ++n) {
        auto dest = xy(out, n*(4+classes+1)*spatical);
        tt::sigmoid(dest, dest);
        dest = probs(out, (n*(4+classes+1) + 4)*spatical);
        tt::sigmoid(dest, dest);
    }
    return input;
}

void YoloLayer::get_yolo_detections(float thresh, std::vector<detection>& dets)
{
    const float *predictions = get_output().host();
    const int width = get_output().nc();
    const int height = get_output().nr();
    const int spatical = width * height;

    for(int n = 0; n < num; ++n) {
        for (int i = 0; i < spatical; ++i) {
            float objectness = predictions[4*spatical + i];
            if(objectness <= thresh) continue;

            // get box
            box b;
            int row = i / width;
            int col = i % width;
            b.x = (col + predictions[i]) / width;
            b.y = (row + predictions[i + spatical]) / height;
            b.w = exp(predictions[i + 2*spatical]) * biases[2*n];
            b.h = exp(predictions[i + 3*spatical]) * biases[2*n+1];

            detection det;
            det.bbox = b;
            det.objectness = objectness;
            det.prob.resize(classes, 0);
            for(int j = 0; j < classes; ++j) {
                float prob = objectness*predictions[(4 + 1 + j)*spatical + i];
                det.prob[j] = (prob > thresh) ? prob : 0;
            }
            dets.push_back(det);
        }
        predictions += (4+classes+1)*spatical;
    }
}


}
