#include "layers.h"

namespace darknet {

ShortcutLayer::ShortcutLayer(LayerPtr _from, int oc, int oh, int ow)
: layer()
, from(_from)
{
    int w2 = from->out_w, h2 = from->out_h, c2 = from->out_c;
    out_w = ow;
    out_h = oh;
    out_c = oc;

    output.set_size(1, this->out_c, this->out_h, this->out_w);

    fprintf(stderr, "res                %4d x%4d x%4d   ->  %4d x%4d x%4d\n", w2,h2,c2, ow,oh,oc);
}

const tensor& ShortcutLayer::forward_layer(const tensor& input)
{
    tt::add(output, input, from->get_output());
    return output;
}


}
