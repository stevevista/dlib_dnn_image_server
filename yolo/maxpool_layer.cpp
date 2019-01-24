#include "layers.h"
#include <cfloat>

namespace darknet {

MaxPoolLayer::MaxPoolLayer(LayerPtr prev, int _size, int _stride)
: window_size(_size)
, stride(_stride)
{
    const int w = prev->get_output().nc();
    const int h = prev->get_output().nr();
    const int k = prev->get_output().k();
    const int out_w = (w - 1)/stride + 1;
    const int out_h = (h - 1)/stride + 1;
    output.set_size(1, k, out_h, out_w);

    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", window_size, window_size, stride, w, h, k, out_w, out_h, k);
}

const tensor& MaxPoolLayer::forward_layer(const tensor& input)
{
    if (stride != 1) {
        mp.setup_max_pooling(window_size, window_size, stride, stride, 0, 0);
        mp(output, input);
    } else {
        int w_offset = -(window_size -1)/2;
        int h_offset = -(window_size -1)/2;

        const int h = output.nr();
        const int w = output.nc();
        const int input_w = input.nc();
        const int input_h = input.nr();
        const int c = input.k();
        auto input_ptr = input.host();
        auto output_ptr = output.host();

        for(int k = 0; k < c; ++k) {
            for(int i = 0; i < h; ++i){
                for(int j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(int n = 0; n < window_size; ++n){
                        for(int m = 0; m < window_size; ++m){
                            int cur_h = h_offset + i*this->stride + n;
                            int cur_w = w_offset + j*this->stride + m;
                            int index = cur_w + input_w*(cur_h + input_h*(k));
                            auto valid = (cur_h >= 0 && cur_h < input_h &&
                                         cur_w >= 0 && cur_w < input_w);
                            float val = valid ? input_ptr[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    output_ptr[out_index] = max;
                }
            }
        }
    }

    return output;
}

}
