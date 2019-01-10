#include "layers.h"
#include <cfloat>

namespace darknet {

MaxPoolLayer::MaxPoolLayer(int c, int h, int w, int _size, int _stride)
: window_size(_size)
, stride(_stride)
{
    this->out_w = (w - 1)/stride + 1;
    this->out_h = (h - 1)/stride + 1;
    this->out_c = c;
    this->output.set_size(1, this->out_c, this->out_h, this->out_w);

    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", window_size, window_size, stride, w, h, c, this->out_w, this->out_h, this->out_c);
}

const tensor& MaxPoolLayer::forward_layer(const tensor& input)
{
    if (stride != 1) {
        mp.setup_max_pooling(window_size, window_size, stride, stride, 0, 0);
        mp(output, input);
        return output;
    } else {
        int w_offset = -(window_size -1)/2;
        int h_offset = -(window_size -1)/2;

        const int h = this->out_h;
        const int w = this->out_w;
        const int input_w = input.nc();
        const int input_h = input.nr();
        const int c = input.k();
        auto input_ptr = input.host();
        auto output_ptr = output.host();

        for(int k = 0; k < c; ++k){
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
