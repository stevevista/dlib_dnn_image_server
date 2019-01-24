#include "layers.h"

namespace darknet {

ConvolutionalLayer::ConvolutionalLayer(LayerPtr prev, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize)
: ConvolutionalLayer(prev->get_output().k(), prev->get_output().nr(), prev->get_output().nc(), n, size, stride, padding,  activation, batch_normalize)
{}

ConvolutionalLayer::ConvolutionalLayer(int c, int h, int w, int filters, int _size, int _stride, int padding, ACTIVATION _activation, int _batch_normalize)
: size(_size)
, stride(_stride)
, pad(padding ? size/2 : 0)
, batch_normalize(_batch_normalize)
, activation(_activation)
{
    const int out_w = (w + 2*this->pad - this->size) / stride + 1;
    const int out_h = (h + 2*this->pad - this->size) / stride + 1;

    beta.set_size(1, filters);
    weights.set_size(filters, c, size, size);
    output.set_size(1, filters, out_h, out_w);

    leaky_param.set_size(1);
    leaky_param = 0.1;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", filters, size, size, stride, w, h, c, out_w, out_h, filters, (2.0 * filters * this->size*this->size*c * out_h*out_w)/1000000000.);
}

void ConvolutionalLayer::load_weights(FILE *fp) {
    if (this->batch_normalize) {
        resizable_tensor sg, sb, running_means, running_variances;
        const int k = output.k();
        sg.set_size(1, k);
        sb.set_size(1, k);
        running_means.set_size(1, k);
        running_variances.set_size(1, k);
        fread(sb.host(), sizeof(float), sb.size(), fp);
        fread(sg.host(), sizeof(float), sg.size(), fp);
        fread(running_means.host(), sizeof(float), k, fp);
        fread(running_variances.host(), sizeof(float), k, fp);

        gamma = pointwise_multiply(mat(sg), 1.0f/sqrt(mat(running_variances) + .000001f));
        beta = mat(sb) - pointwise_multiply(mat(gamma), mat(running_means));
    } else {
        fread(beta.host(), sizeof(float), beta.size(), fp);
    }
    fread(weights.host(), sizeof(float), weights.size(), fp);
}

const tensor& ConvolutionalLayer::forward_layer(const tensor& input)
{
    conv.setup(input,
        weights,
        stride,
        stride,
        pad,
        pad);
    conv(false, output,
        input,
        weights);

    if(this->batch_normalize) {
        tt::affine_transform_conv(output, output, gamma, beta);
    } else {
        tt::add(1,output,1, beta);
    }

    if (activation == LEAKY) {
        tt::prelu(output, output, leaky_param);
    }

    return output;
}

}
