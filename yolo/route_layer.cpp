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
    int offset = 0;
    for(auto in : input_layers) {
        const float *input = in->get_output().host();
        int input_size = in->get_output().size();
        memcpy(output.host() + offset, input, sizeof(float) * input_size);
        offset += input_size;
    }
    return output;
}


}
