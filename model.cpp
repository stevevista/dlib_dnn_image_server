#include "model.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include "yolo/yolo.h"
#include "landmark.hpp"


// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// mmod face CNN model
// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using mmod_net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


// ----------------------------------------------------------------------------------------

inline std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

// ----------------------------------------------------------------------------------------

shape_predictor sp;
anet_type net;
frontal_face_detector detector;
mmod_net_type mmod_net;
darknet::NetworkPtr yolonet;


model::model(const string& _basedir, int landmarks, bool _use_mmod, long upsize, int _yolo_type) 
: basedir(_basedir)
, face_landmarks(landmarks)
, use_mmod(_use_mmod)
, pyramid_upsize(upsize)
, yolo_type(_yolo_type)
{
  detector = get_frontal_face_detector();

  deserialize(face_landmarks == 68 ? basedir + "shape_predictor_68_face_landmarks.dat" : basedir + "shape_predictor_5_face_landmarks.dat") >> sp;
  // And finally we load the DNN responsible for face recognition.
  deserialize(basedir + "dlib_face_recognition_resnet_model_v1.dat") >> net;

  if (use_mmod) deserialize(basedir + "mmod_human_face_detector.dat") >> mmod_net;

  if (yolo_type == 1) yolonet = darknet::loadYoloNet(basedir, basedir + "yolov3-tiny.weights");
  else if (yolo_type == 2) yolonet = darknet::loadYoloNet(basedir, basedir + "yolov3.weights");
}

std::vector<rectangle> model::detect_faces(matrix<rgb_pixel>& img) {
    std::vector<rectangle> dets;
    if (use_mmod) {
      while(img.size() < pyramid_upsize*pyramid_upsize)
        pyramid_up(img);
      auto mmdets = mmod_net(img);
      for (auto& mm : mmdets) {
        dets.push_back(mm.rect);
      }
    } else {
        dets = detector(img);

      if (dets.size() == 0) {
                matrix<rgb_pixel> dimg;
                rotate_image(img, dimg, -pi/2);
                dets = detector(dimg);
                if (dets.size()) {
                    img = dimg;
                }
      }

      if (dets.size() == 0) {
        matrix<rgb_pixel> dimg;
        rotate_image(img, dimg, pi/2);
        dets = detector(dimg);
        if (dets.size()) {
          img = dimg;
        }
      }
    }

    return dets;
}

int model::predict_objects(matrix<rgb_pixel>& img, float thresh, float nms) {
  if (!yolonet) {
    return -1;
  }
  
  auto start_time = std::chrono::system_clock::now();
  auto dets = yolonet->predict_yolo(img, thresh, nms);
  auto eclipsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();
  cout << "Predicted in " << eclipsed << " millisecs" << endl;

  return draw_detections(img, dets);
}

std::vector<face_dectection> model::predict_faces(matrix<rgb_pixel>& img, int mark_thickness, rgb_pixel mark_color) {
  auto dets = detect_faces(img);

  if (dets.size() == 0) {
    return {};
  }

  std::vector<matrix<rgb_pixel>> faces;
  std::vector<full_object_detection> shapes;
  for (auto face : dets) {
    auto shape = sp(img, face);
    matrix<rgb_pixel> face_chip;
    extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
    faces.push_back(move(face_chip));
    shapes.push_back(move(shape));
  }

  // This call asks the DNN to convert each face image in faces into a 128D vector.
  // In this 128D vector space, images from the same person will be close to each other
  // but vectors from different people will be far apart.  So we can use these vectors to
  // identify if a pair of images are from the same person or from different people.  
  auto start_time = std::chrono::system_clock::now();
  mtx.lock();
  auto face_descriptors = net(faces);
  mtx.unlock();
  auto eclipsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();
  cout << "predict " << face_descriptors.size() << "  faces in " << eclipsed << " millisecs" << endl;

  std::vector<face_dectection> results;
  for (auto i=0; i<dets.size(); i++) {
    if (mark_thickness > 0) {
      draw_face_landmark(img, shapes[i], mark_color, mark_thickness);
    }

    results.push_back({move(dets[i]), move(faces[i]), move(face_descriptors[i]), move(shapes[i])});
  }

  return results;
}


std::vector<face_dectection> model::filter_faces(const std::vector<face_dectection>& dets, const matrix<float,0,1>& target, float thres, bool match_all) {

  std::vector<face_dectection> filtered;
  float min_delta = thres;
  int selected = -1;

  for (size_t i = 0; i < dets.size(); ++i) {
    auto delta = length(target - dets[i].descriptor);
    if (match_all) {
      if (delta < thres) {
        filtered.push_back(move(dets[i]));
      }
    } else {
      if (delta < min_delta) {
        selected = i;
        min_delta = delta;
      }
    }
  }

  if (selected >= 0) {
    filtered.push_back(move(dets[selected]));
  }

  return filtered;
}
