#include "layers.h"

namespace darknet {

UpSampleLayer::UpSampleLayer(int c, int h, int w, int _stride)
: stride(_stride)
{
    this->out_w = w*stride;
    this->out_h = h*stride;
    this->out_c = c;
    this->output.set_size(1, this->out_c, this->out_w, this->out_h);

    fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, this->out_w, this->out_h, this->out_c);
}

const tensor& UpSampleLayer::forward_layer(const tensor& input)
{
    output = 0;
    float *out = output.host();
    const float *in = input.host();
    const int h = input.nr();
    const int w = input.nc();

    for(int k = 0; k < input.k(); ++k) {
            for(int j = 0; j < h*stride; ++j){
                for(int i = 0; i < w*stride; ++i){
                    int in_index = k*w*h + (j/stride)*w + i/stride;
                    int out_index = k*w*h*stride*stride + j*w*stride + i;
                    out[out_index] = in[in_index];
                }
            }
    }
    return output;
}


}

