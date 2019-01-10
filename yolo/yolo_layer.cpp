#include "layers.h"

namespace darknet {

YoloLayer::YoloLayer(int h, int w, int _classes, const std::vector<float>& _biases) 
: num(_biases.size()/2)
, classes(_classes)
, biases(_biases)
, output_ptr(nullptr)
{
    this->out_w = w;
    this->out_h = h;
    this->out_c = num*(classes + 4 + 1);

    xy = alias_tensor(2*w*h);
    probs = alias_tensor((classes+1)*w*h);

    fprintf(stderr, "yolo\n");
}

const tensor& YoloLayer::forward_layer(const tensor& input)
{
    float *X = const_cast<float*>(input.host());
    tensor& out = const_cast<tensor&>(input);

    const int spatical = out_h * out_w;

    for(int n = 0; n < num; ++n) {
        auto dest = xy(out, n*(4+classes+1)*spatical);
        tt::sigmoid(dest, dest);
        dest = probs(out, (n*(4+classes+1) + 4)*spatical);
        tt::sigmoid(dest, dest);
    }
    output_ptr = &input;
    return input;
}

void YoloLayer::get_yolo_detections(float thresh, std::vector<detection>& dets)
{
    if (!output_ptr) {
        return;
    }
    const float *predictions = output_ptr->host();
    const int width = out_w;
    const int height = out_h;
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
