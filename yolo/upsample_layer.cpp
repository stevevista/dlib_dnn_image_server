#include "layers.h"

namespace darknet {

UpSampleLayer::UpSampleLayer(LayerPtr prev, int stride)
{
    const int out_w = prev->get_output().nc()*stride;
    const int out_h = prev->get_output().nr()*stride;
    const int k = prev->get_output().k();
    output.set_size(1, k, out_h, out_w);

    fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, prev->get_output().nc(), prev->get_output().nr(), k, out_w, out_h, k);
}

const tensor& UpSampleLayer::forward_layer(const tensor& input)
{
    tt::resize_bilinear(output, input);
    return output;
}


}

