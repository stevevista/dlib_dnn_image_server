#include <time.h>
#include <assert.h>
#include "yolo.h"
#include "layers.h"


namespace darknet {

const std::vector<std::string> coco_names = {
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
};

void makeYoloLayers(network* net, int classes, const std::vector<float>& biases) {
    const int filters = (biases.size()/2)*(classes + 4 + 1);
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters, 1, 1, 1, LINEAR, 0));
    net->add_layer(std::make_shared<YoloLayer>(net->back(), biases));
}

void makeShortcuts(network* net, int filters) {
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters, 1));
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters*2, 3));
    net->add_layer(std::make_shared<ShortcutLayer>(net->layer(-3), net->back()));
}

void makeDownSamples(network* net, int filters, int repeats) {
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters*2, 3, 2));
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters, 1));
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters*2, 3));
    net->add_layer(std::make_shared<ShortcutLayer>(net->layer(-3), net->back()));

    for (int i = 0; i < repeats; i++) {
        makeShortcuts(net, filters);
    }
}

void makeUpSamples(network* net, int filters, int route_index) {
    net->add_layer(std::make_shared<RouteLayer>(std::vector<LayerPtr>{net->layer(-4)}));
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters, 1));
    net->add_layer(std::make_shared<UpSampleLayer>(net->back(), 2));
    net->add_layer(std::make_shared<RouteLayer>(std::vector<LayerPtr>{net->layer(-1), net->layer(route_index)}));
}

void repeatConvolutionalLayers(network* net, int filters, int repeats) {
    for (int i = 0; i < repeats; i++) {
        net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters, 1));
        net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters*2, 3));
    }
}

NetworkPtr YoloSPPNet(const std::string& data_dir)
{
    auto net = std::make_shared<network>();

    net->w  = 608; 
    net->h = 608;

    net->add_layer(std::make_shared<ConvolutionalLayer>(3, net->h, net->w, 32, 3));
    // Downsample
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), 64, 3, 2));
    makeShortcuts(net.get(), 32);
    // Downsample
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), 128, 3, 2));
    makeShortcuts(net.get(), 64);
    makeShortcuts(net.get(), 64);
    // Downsample
    makeDownSamples(net.get(), 128, 7);
    makeDownSamples(net.get(), 256, 7);
    makeDownSamples(net.get(), 512, 3);

    repeatConvolutionalLayers(net.get(), 512, 1);
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), 512, 1));
    // SPP
    net->add_layer(std::make_shared<MaxPoolLayer>(net->back(), 5, 1));
    net->add_layer(std::make_shared<RouteLayer>(std::vector<LayerPtr>{net->layer(-2)}));
    net->add_layer(std::make_shared<MaxPoolLayer>(net->back(), 9, 1));
    net->add_layer(std::make_shared<RouteLayer>(std::vector<LayerPtr>{net->layer(-4)}));
    net->add_layer(std::make_shared<MaxPoolLayer>(net->back(), 13, 1));
    net->add_layer(std::make_shared<RouteLayer>(std::vector<LayerPtr>{net->layer(-1), net->layer(-3), net->layer(-5), net->layer(-6)}));
    // End SPP
    repeatConvolutionalLayers(net.get(), 512, 2);
    makeYoloLayers(net.get(), 500, std::vector<float>{116,90,  156,198,  373,326});

    makeUpSamples(net.get(), 256, 61);

    repeatConvolutionalLayers(net.get(), 256, 3);
    makeYoloLayers(net.get(), 500, std::vector<float>{30,61,  62,45,  59,119});

    makeUpSamples(net.get(), 128, 36);

    repeatConvolutionalLayers(net.get(), 128, 3);
    makeYoloLayers(net.get(), 500, std::vector<float>{10,13,  16,30,  33,23});

    load_alphabets(data_dir);
    return net;
}

NetworkPtr YoloNet(const std::string& data_dir)
{
    auto net = std::make_shared<network>();

    net->w  = 608; 
    net->h = 608;

    net->add_layer(std::make_shared<ConvolutionalLayer>(3, net->h, net->w, 32, 3));
    // Downsample
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), 64, 3, 2));
    makeShortcuts(net.get(), 32);
    // Downsample
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), 128, 3, 2));
    makeShortcuts(net.get(), 64);
    makeShortcuts(net.get(), 64);
    // Downsample
    makeDownSamples(net.get(), 128, 7);
    makeDownSamples(net.get(), 256, 7);
    makeDownSamples(net.get(), 512, 3);

    repeatConvolutionalLayers(net.get(), 512, 3);
    makeYoloLayers(net.get(), 80, std::vector<float>{116,90,  156,198,  373,326});

    makeUpSamples(net.get(), 256, 61);
    repeatConvolutionalLayers(net.get(), 256, 3);
    makeYoloLayers(net.get(), 80, std::vector<float>{30,61,  62,45,  59,119});

    makeUpSamples(net.get(), 128, 36);
    repeatConvolutionalLayers(net.get(), 128, 3);
    makeYoloLayers(net.get(), 80, std::vector<float>{10,13,  16,30,  33,23});
    
    load_alphabets(data_dir);
    return net;
}

NetworkPtr TinyYoloNet(const std::string& data_dir) {
    auto net = std::make_shared<network>();

    net->w  = 416; 
    net->h = 416;

    net->add_layer(std::make_shared<ConvolutionalLayer>(3, net->h, net->w, 16, 3));
    net->add_layer(std::make_shared<MaxPoolLayer>(net->back(), 2, 2));

    int filters = 32;
    for (int i = 0; i < 4; i++, filters *= 2) {
        net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), filters, 3));
        net->add_layer(std::make_shared<MaxPoolLayer>(net->back(), 2, 2));
    }

    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), 512, 3));
    net->add_layer(std::make_shared<MaxPoolLayer>(net->back(), 2, 1));
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), 1024, 3));

    repeatConvolutionalLayers(net.get(), 256, 1);
    makeYoloLayers(net.get(), 80, std::vector<float>{81,82,  135,169,  344,319});

    makeUpSamples(net.get(), 128, 8);
    net->add_layer(std::make_shared<ConvolutionalLayer>(net->back(), 256, 3));
    makeYoloLayers(net.get(), 80, std::vector<float>{10,14,  23,27,  37,58});

    load_alphabets(data_dir);
    return net;
}

NetworkPtr loadYoloNet(const std::string& data_dir, const std::string& filename) {

    NetworkPtr net;
    if (filename.find("tiny.weights") != std::string::npos) {
        net = TinyYoloNet(data_dir);
    } else if (filename.find("spp_final.weights") != std::string::npos) {
        net = YoloSPPNet(data_dir);
    } else {
        net = YoloNet(data_dir);
    }

    net->load_weights(filename);
    return net;
}

LayerPtr network::back() {
    return layers.back();
}

LayerPtr network::layer(int offset) {
    if (offset < 0) return layers[layers.size() + offset];
    else return layers[offset];
}

void network::load_weights(const std::string& filename)
{
    fprintf(stderr, "Loading weights from %s...", filename.c_str());
    fflush(stdout);
    FILE *fp = fopen(filename.c_str(), "rb");
    if(!fp) throw std::runtime_error("Loading weights error");

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        size_t iseen = 0;
        fread(&iseen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
    }
    for(auto& l : layers) {
        l->load_weights(fp);
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void network::predict(const tensor& x)
{
    const tensor* pinput = &x;
    for(auto& l : layers) {
        pinput = &l->forward_layer(*pinput);
    }
}

void do_nms(std::vector<detection>& dets, float thresh);

std::vector<detection> network::predict_boxes(int img_w, int img_h, int new_w, int new_h, float thresh, float nms)
{
    const int netw = this->w;
    const int neth = this->h;

    std::vector<detection> dets;

    mtx.lock();
    predict(input);

    for(auto& l : layers) {
        l->get_yolo_detections(thresh, dets);
    }
    mtx.unlock();

    for (auto& det : dets) {
        // correct box
        det.bbox.x = (det.bbox.x - 0.5)*netw + 0.5*new_w;
        det.bbox.y = (det.bbox.y - 0.5)*neth + 0.5*new_h;
    }

    do_nms(dets, nms);

    for (auto& det : dets) {
        box b = det.bbox;
        int left  = (b.x-b.w/2.)*img_w/new_w;
        int right = (b.x+b.w/2.)*img_w/new_w;
        int top   = (b.y-b.h/2.)*img_h/new_h;
        int bot   = (b.y+b.h/2.)*img_h/new_h;

        if(left < 0) left = 0;
        if(right > img_w-1) right = img_w-1;
        if(top < 0) top = 0;
        if(bot > img_h-1) bot = img_h-1;
        det.rect = rectangle(left, top, right, bot);
    }

    return dets;
}

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

static float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

static float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void do_nms(std::vector<detection>& dets, float thresh)
{
    const int total = dets.size();
    const int classes = total ? dets[0].prob.size() : 0;

    for(int k = 0; k < classes; ++k) {
        std::sort(dets.begin(), dets.end(), [=](const detection& a, const detection& b) { return a.prob[k] > b.prob[k]; });
        for(int i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            dets[i].candicates.push_back(k);
            box a = dets[i].bbox;
            for(int j = i+1; j < total; ++j) {
                if(dets[j].prob[k] == 0) continue;
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }

    dets.erase(std::remove_if(dets.begin(), 
                              dets.end(),
                              [](const detection& a){ return a.candicates.empty(); }),
               dets.end());
}

int draw_detections(matrix<dlib::rgb_pixel>& im, const std::vector<detection>& dets) {
    return draw_detections(im, dets, coco_names);
}

}

