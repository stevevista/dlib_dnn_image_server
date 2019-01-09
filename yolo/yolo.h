#pragma once
#include <vector>
#include <memory>

#include <dlib/cuda/tensor_tools.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>

namespace darknet {

using namespace dlib;

struct network;
struct  layer;

typedef std::shared_ptr<layer> LayerPtr; 
typedef std::shared_ptr<network> NetworkPtr; 

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection {
    box bbox;
    std::vector<float> prob;
    float objectness;
    int sort_class;
} detection;

struct layer {
    virtual ~layer() {}
    virtual void load_weights(FILE *fp) {}
    virtual const tensor& forward_layer(const tensor& net) = 0;
    virtual const tensor& get_output() const = 0;
    virtual void get_yolo_detections(float thresh, std::vector<detection>& dets) { }

    int out_h, out_w, out_c;
};

struct network {

    LayerPtr back();
    std::vector<detection> get_boxes(int w, int h, float thresh);

    void load_weights(const char *filename);
    void predict(const tensor& x);

    template<typename image_type>
    std::vector<detection> predict_yolo(const image_type& src_img, float thresh) {
        auto img = src_img;
        double scale;
        if (((double)this->w/img.nc()) < ((double)this->h/img.nr())) {
            scale = (double)this->w/img.nc();
        } else {
            scale = (double)this->h/img.nr();
        }

        dlib::resize_image(scale, img);

        int new_w = img.nc();
        int new_h = img.nr();
        int w_offset = (w-new_w)/2;
        int h_offset = (h-new_h)/2;

        input.set_size(1, 3, h, w);
        const size_t offset = this->h*this->w;
        auto ptr = input.host();

        for (int r = 0; r < this->h; ++r) {
            for (int c = 0; c < this->w; ++c) {
                auto p = ptr++;
                if (c < w_offset || c >= (w_offset + new_w) ||
                        r < h_offset || r >= (h_offset + new_h)) {
                    *p = 0.5; 
                    p += offset;
                    *p = 0.5; 
                    p += offset;
                    *p = 0.5; 
                } else {
                    rgb_pixel temp = img(r-h_offset,c-w_offset);
                    *p = (temp.red)/255.0; 
                    p += offset;
                    *p = (temp.green)/255.0; 
                    p += offset;
                    *p = (temp.blue)/255.0; 
                }
            }
        }

        predict(input);
        return get_boxes(src_img.nc(), src_img.nr(), thresh);
    }

    int h, w;
    std::vector<LayerPtr> layers;
    resizable_tensor input;
} ;


NetworkPtr YoloNet();
NetworkPtr TinyYoloNet();
NetworkPtr YoloSPPNet();
NetworkPtr loadYoloNet(const std::string& filename);

int draw_detections(matrix<dlib::rgb_pixel>& im, const std::vector<detection>& dets, float thresh, const std::vector<std::string>& names);
int draw_detections(matrix<dlib::rgb_pixel>& im, const std::vector<detection>& dets, float thresh);

extern std::string runtime_dir;
void load_alphabets();

void do_nms_obj(std::vector<detection>& dets, float thresh);
void do_nms_sort(std::vector<detection>& dets, float thresh);

}

