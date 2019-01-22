#include "yolo.h"
#include <dlib/matrix.h>

namespace darknet {

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

static unsigned char get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r * 255;
}

////////
std::vector<matrix<dlib::rgb_pixel>> alphabets(128);

void load_alphabets(const std::string& data_dir)
{
    static int inited = 0;
    if (inited) return;

    inited = 1;

    for (int i =32; i < 127; i++) {
        char buff[256];
        sprintf(buff, "data/labels/%d_3.png", i);
        matrix<dlib::rgb_pixel> img;
        load_image(img,  data_dir + buff);
        alphabets[i] = img;
    }
}

void composite_image(const matrix<dlib::rgb_pixel>& source, matrix<dlib::rgb_pixel>& dest, int dx, int dy)
{
    for(int y = 0; y < source.nr(); ++y) {
        for(int x = 0; x < source.nc(); ++x) {
            if(dx+x >= 0 && dx+x < dest.nc() && y + dy >= 0 && y + dy < dest.nr()) {
                auto val = source(y, x);
                auto val2 = dest(y+dy, dx+x);
                val.red = (val.red * val2.red) / (255);
                val.green = (val.green * val2.green) / (255);
                val.blue = (val.blue * val2.blue) / (255);
                dest(y+dy, dx+x) = val;
            }
        }
    }
}

void draw_label(matrix<dlib::rgb_pixel>& a, int r, int c, const char *string, rgb_pixel rgb)
{
    int size = a.nr()*.003;
    if(size > 3) size = 3;
    double scale = ((double)(size+1)) /4;
    auto interp = interpolate_bilinear();
    
    std::vector<matrix<dlib::rgb_pixel>> chars;
    while(*string) {
        auto l = alphabets[(int)*string];
        if (!l.nr()) l = alphabets[(int)' '];

        matrix<dlib::rgb_pixel> c;
        c.set_size(l.nr()*scale, l.nc()*scale);

        resize_image(l, c, interp);
        chars.push_back(c);
        ++string;
    }

    int dx = -size - 1 + (size+1)/2;
    int height = 0;
    int width = 0;
    for (const auto& c : chars) {
        if (height < c.nr()) {
            height = c.nr();
        }
        width += c.nc() + dx;
    }

    int border = height*.25;
    int w = width + 2*border;
    int h = height + 2*border;

    if (r - h >= 0) r = r - h;

    for (int j=0; j<h; j++) {
        for (int i=0; i<w; i++) {
            if (r+j >=0 && r+j < a.nr() && c+i >=0 && c+i < a.nc())
                a(r+j, c+i) = rgb;
        }
    }

    int nc = border;
    for (const auto& ci : chars) {
        composite_image(ci, a, nc + dx + c, border + r);
        nc += ci.nc() + dx;
    }
}

template <
        typename image_type,
        typename pixel_type
        >
void draw_labled_rectangle(image_type& img, const rectangle& rect, const std::string& string, const pixel_type& val, unsigned int thickness) {
    draw_rectangle(img, rect, val, thickness);
    draw_label(img, rect.top() + thickness, rect.left(), string.c_str(), val);
}

int draw_detections(matrix<dlib::rgb_pixel>& im, const std::vector<detection>& dets, const std::vector<std::string>& names)
{
    const int classes = dets.size() ? dets[0].prob.size() : 0;

    for(auto& det : dets) {
        std::string labelstr;
        for (auto cls : det.candicates) {
            std::string label = cls < names.size() ? names[cls] : std::to_string(cls);
            if (labelstr.size()) {
                labelstr += ", ";
            }
            labelstr += label;
            printf("%s: %.0f%%\n", label.c_str(), det.prob[cls]*100);
        }

        const int thickness = im.nr() * .006;

        rgb_pixel clr;
        int offset = det.candicates[0]*123457 % classes;
        clr.red = get_color(2,offset,classes);
        clr.green = get_color(1,offset,classes);
        clr.blue = get_color(0,offset,classes);

        draw_labled_rectangle(im, det.rect, labelstr, clr, thickness);
    }

    return dets.size();
}

}
