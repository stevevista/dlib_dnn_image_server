#pragma once
#include <dlib/image_processing/full_object_detection.h>

using namespace dlib;
using namespace std;

template <
        typename image_type,
        typename pixel_type
        >
void draw_face_landmark(
  image_type& img,
  full_object_detection& det,
  const pixel_type& pixel,
  unsigned int thickness = 1
) {

  auto line = [&](int start, int end) {
    draw_line(det.part(start).x(), det.part(start).y(), det.part(end).x(), det.part(end).y(), img, pixel);
    for (unsigned int i = 1; i < thickness; i++)
      draw_line(det.part(start).x(), det.part(start).y() + i, det.part(end).x(), det.part(end).y() + i, img, pixel);
  };

  if (det.num_parts() == 5) {
    line(0, 1);
    line(1, 4);
    line(4, 3);
    line(3, 2);
  } else if (det.num_parts() == 68) {
    // Around Chin. Ear to Ear
    for (unsigned long i = 1; i <= 16; ++i) {
      line(i, i-1);
    }

    // Line on top of nose
    for (unsigned long i = 28; i <= 30; ++i)
      line(i, i-1);

    // left eyebrow
    for (unsigned long i = 18; i <= 21; ++i)
      line(i, i-1);

    // Right eyebrow
    for (unsigned long i = 23; i <= 26; ++i)
      line(i, i-1);

    // Bottom part of the nose
    for (unsigned long i = 31; i <= 35; ++i)
      line(i, i-1);

    // Line from the nose to the bottom part above
    line(30, 35);
            
    for (unsigned long i = 37; i <= 41; ++i)
      line(i, i-1);
    line(36, 41);

    // Right eye
    for (unsigned long i = 43; i <= 47; ++i)
      line(i, i-1);
    line(42, 47);

    // Lips outer part
    for (unsigned long i = 49; i <= 59; ++i)
      line(i, i-1);
    line(48, 59);

    // Lips inside part
    for (unsigned long i = 61; i <= 67; ++i)
      line(i, i-1);
    line(60, 67);
  }
}
