
cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(examples)

# -DUSE_SSE4_INSTRUCTIONS=ON -DUSE_AVX_INSTRUCTIONS=ON -DDLIB_NO_GUI_SUPPORT=ON

# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
add_subdirectory(../dlib dlib_build)

add_executable(image_server image_server.cpp
         server.cpp
         model.cpp
         utils.cpp
         multipart.cpp
         yolo/convolutional_layer.cpp
         yolo/image.cpp
         yolo/maxpool_layer.cpp
         yolo/network.cpp
         yolo/route_layer.cpp
         yolo/upsample_layer.cpp
         yolo/box.cpp
         yolo/shortcut_layer.cpp 
         yolo/yolo_layer.cpp)
target_link_libraries(image_server dlib::dlib )
