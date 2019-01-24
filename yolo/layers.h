#pragma once
#include "yolo.h"

namespace darknet {

typedef enum {
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

class ConvolutionalLayer : public layer {
  const int size;
  const int stride;
  const int pad;
  const int batch_normalize;

  ACTIVATION activation;
  resizable_tensor weights;
  resizable_tensor gamma;
  resizable_tensor beta;

  tt::tensor_conv conv;
  resizable_tensor output;
  resizable_tensor leaky_param;

public:
  ConvolutionalLayer(LayerPtr prev, int n, int size, int stride = 1, int padding = 1, ACTIVATION activation = LEAKY, int batch_normalize = 1);
  ConvolutionalLayer(int c, int h, int w, int n, int size, int stride = 1, int padding = 1, ACTIVATION activation = LEAKY, int batch_normalize = 1);
  const tensor& forward_layer(const tensor& net);
  const tensor& get_output() const { return output; }
  void load_weights(FILE *fp);
};

class MaxPoolLayer : public layer {
  const int window_size;
  const int stride;
  resizable_tensor output;
  tt::pooling mp;
public:
  MaxPoolLayer(LayerPtr prev, int size, int stride);
  const tensor& forward_layer(const tensor& net);
  const tensor& get_output() const { return output; }
};

class RouteLayer : public layer {
  std::vector<LayerPtr> input_layers;
  resizable_tensor output;
public:
  RouteLayer(std::vector<LayerPtr> inputs);
  const tensor& forward_layer(const tensor& net);
  const tensor& get_output() const { return output; }
};

class ShortcutLayer : public layer {
  LayerPtr from;
  resizable_tensor output;
public:
  ShortcutLayer(LayerPtr from, LayerPtr prev);
  const tensor& forward_layer(const tensor& net);
  const tensor& get_output() const { return output; }
};

class UpSampleLayer : public layer {
  resizable_tensor output;
public:
  UpSampleLayer(LayerPtr prev, int stride);
  const tensor& forward_layer(const tensor& net);
  const tensor& get_output() const { return output; }
};

class YoloLayer : public layer {
  LayerPtr linkin;
  const int num;
  int classes;
  const std::vector<float> biases;
  alias_tensor xy, probs;
public:
  YoloLayer(LayerPtr linkin, const std::vector<float>& biases);
  const tensor& forward_layer(const tensor& net);
  const tensor& get_output() const { return linkin->get_output(); }
  void get_yolo_detections(float thresh, std::vector<detection>& dets);
};

}
