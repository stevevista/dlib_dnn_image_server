#include "layers.h"

namespace darknet {

UpSampleLayer::UpSampleLayer(int c, int h, int w, int stride)
{
    this->out_w = w*stride;
    this->out_h = h*stride;
    out_c = c;
    this->output.set_size(1, out_c, this->out_h, this->out_w);

    fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, this->out_w, this->out_h, out_c);
}

const tensor& UpSampleLayer::forward_layer(const tensor& input)
{
    tt::resize_bilinear(output, input);
    return output;
}


}

