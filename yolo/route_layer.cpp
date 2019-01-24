#include "layers.h"

namespace darknet {

RouteLayer::RouteLayer(std::vector<LayerPtr> inputlayers)
: input_layers(inputlayers)
{
    auto first = inputlayers[0];
    const int out_w = first->get_output().nc();
    const int out_h = first->get_output().nr();
    int out_k = first->get_output().k();
    for(auto i = 1; i < inputlayers.size(); ++i) {
        auto next = inputlayers[i];
        DLIB_CASSERT(next->get_output().nr() == out_h && next->get_output().nc() == out_w);
        out_k += next->get_output().k();
    }

    output.set_size(1, out_k, out_h, out_w);
    fprintf(stderr,"route \n");
}

const tensor& RouteLayer::forward_layer(const tensor&)
{
    size_t k_offset = 0;
    for(auto in : input_layers) {
        size_t count_k = in->get_output().k();
        tt::copy_tensor(
            false,
            output,
            k_offset,
            in->get_output(),
            0,
            count_k
        );
        k_offset += count_k;
    }
    return output;
}


}
