#include "layers.h"

namespace darknet {

RouteLayer::RouteLayer(std::vector<LayerPtr> inputlayers)
: input_layers(inputlayers)
{
    auto first = inputlayers[0];
    out_w = first->out_w;
    out_h = first->out_h;
    out_c = first->out_c;
    for(auto i = 1; i < inputlayers.size(); ++i) {
        auto next = inputlayers[i];
        if(next->out_w == first->out_w && next->out_h == first->out_h) {
            out_c += next->out_c;
        }else{
            out_h = out_w = out_c = 0;
        }
    }

    output.set_size(1, out_c, out_h, out_w);
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
