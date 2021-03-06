#include "layers.h"

namespace darknet {

ShortcutLayer::ShortcutLayer(LayerPtr _from, LayerPtr prev)
: from(_from)
{
    output.copy_size(prev->get_output());

    fprintf(stderr, "res                %4d x%4d x%4d   ->  %4d x%4d x%4d\n", (int)from->get_output().nc(), (int)from->get_output().nr(), (int)from->get_output().k(), (int)output.nc(), (int)output.nr(), (int)output.k());
}

const tensor& ShortcutLayer::forward_layer(const tensor& input)
{
    tt::add(output, input, from->get_output());
    return output;
}


}
