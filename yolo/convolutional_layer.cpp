#include "layers.h"

namespace darknet {

ConvolutionalLayer::ConvolutionalLayer(int c, int h, int w, int n, int _size, int _stride, int padding, ACTIVATION _activation, int _batch_normalize)
: size(_size)
, stride(_stride)
, pad(padding ? size/2 : 0)
, batch_normalize(_batch_normalize)
, activation(_activation)
{
    out_w = (w + 2*this->pad - this->size) / this->stride + 1;
    out_h = (h + 2*this->pad - this->size) / this->stride + 1;
    out_c = n;

    beta.set_size(1, out_c);
    filters.set_size(out_c, c, size, size);

    if (batch_normalize) {
        gamma.set_size(1, out_c);
        running_means.set_size(1, out_c);
        running_variances.set_size(1, out_c);
    }

    output.set_size(1, out_c, out_h, out_w);

    leaky_param.set_size(1);
    leaky_param = 0.1;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, this->out_w, this->out_h, this->out_c, (2.0 * this->out_c * this->size*this->size*c * this->out_h*this->out_w)/1000000000.);
}

void ConvolutionalLayer::load_weights(FILE *fp) {
    fread(beta.host(), sizeof(float), beta.size(), fp);
    if (this->batch_normalize){
        fread(this->gamma.host(), sizeof(float), this->out_c, fp);
        fread(this->running_means.host(), sizeof(float), this->out_c, fp);
        fread(this->running_variances.host(), sizeof(float), this->out_c, fp);
    }
    fread(filters.host(), sizeof(float), filters.size(), fp);
}

const tensor& ConvolutionalLayer::forward_layer(const tensor& input)
{
    conv.setup(input,
        filters,
        stride,
        stride,
        pad,
        pad);
    conv(false, output,
        input,
        filters);

    if(this->batch_normalize) {
        tt::batch_normalize_conv_inference(.000001f, output, output, gamma, beta, running_means, running_variances);
    } else {
        tt::add(1,output,1, beta);
    }

    if (activation == LEAKY) {
        tt::prelu(output, output, leaky_param);
    }

    return output;
}

}
