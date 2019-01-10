#include <iostream>
#include <sstream>
#include <string>
#include <dlib/config_reader.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include "landmark.hpp"
#include "utils.h"
#include "server.h"
#include "model.h"

/*
spec:
detect_face
detect_object

*/
float thres = 0.6;
int box = 3;
int mark = 3;
int png = 0;

class image_server : public web_server {

    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    model modeler;

protected:
    void on_post_detect(std::ostream& out, const incoming_things&, outgoing_things&);
    void on_post_resize(std::ostream& out, const incoming_things&, outgoing_things&);

    int detect_face(ostringstream& sout, 
                matrix<dlib::rgb_pixel>& img,
                const string& basedir,
                const matrix<float,0,1>& feature);

public:
    image_server(const string& basedir, int landmarks, bool use_mmod, long upsize, int yolo_type);
    void install_routes();
};

image_server::image_server(const string& basedir, int landmarks, bool use_mmod, long upsize, int yolo_type)
: modeler(basedir, landmarks, use_mmod, upsize, yolo_type) {
        
}

int image_server::detect_face(ostringstream& sout, 
                matrix<dlib::rgb_pixel>& img,
                const string& basedir,
                const matrix<float,0,1>& feature) {

    std::vector<face_dectection> dets;
    dets = modeler.predict_faces(img, mark);

    if (dets.size() == 0) {
        cout << "No face detected" << endl;
        return modeler.predict_objects(img, .5, .45);
    }

    // filter
    if (feature.size()) {
        dets = modeler.filter_faces(dets, feature, thres, false);
        if (dets.empty()) {
            return 0;
        }
    }

    sout << "\"detects\": [";

    for (size_t i = 0; i < dets.size(); ++i) {
        auto sub_path = random_string() + (png ? ".png" : ".jpg");

        if (!png) save_jpeg(dets[i].face, basedir + sub_path);
        else save_png(dets[i].face, basedir + sub_path);

        if (box > 0)
            draw_rectangle(img, dets[i].rect, rgb_pixel(0, 255, 0), box);

        if (i > 0) sout << ",";
        sout << "{\"path\":\"" << sub_path << "\"";
        sout << ",\"desc\":[";
        auto& desc = dets[i].descriptor;
        for (int n = 0; n < desc.size(); n++) {
            if (n > 0) sout << ",";
            sout << desc(n);
        }
        sout << "]";
        sout << "}";
    }

    sout << "],";

    return dets.size();
}

void image_server::on_post_detect(std::ostream& out, const incoming_things& incoming, outgoing_things& outgoing) {
    
    ostringstream sout;
    auto upload_path = incoming.queries["files/file/path"];
    auto filepath = incoming.queries["filepath"];
    auto spec = incoming.queries["spec"];

    if (upload_path.size())
        filepath = upload_path;

    matrix<rgb_pixel> img;
    load_image(img,  filepath);

    auto basedir = get_dirname(filepath);
    auto output_name = random_string() + (png ? ".png" : ".jpg");
    int detects = 0;

    outgoing.headers["content-type"] = "application/json";

    sout << "{";

    if (spec == "detect_object") {
        detects = modeler.predict_objects(img, .5, .45);
    } else {
        auto bench_data = split_float_array(spec);
        if (bench_data.size() == 128) {

            matrix<float,0,1> feature;
            feature.set_size(128);
            for (int i=0; i< feature.size(); i++) {
                feature(i) = bench_data[i];
            }
            detects = detect_face(sout, 
                img,
                basedir,
                feature);
        } else {
            detects = detect_face(sout, 
                img,
                basedir,
                matrix<float,0,1>());
        }
    }

    if (detects > 0) {
        if (!png) save_jpeg(img, basedir + output_name);
        else save_png(img, basedir + output_name);

        sout << "\"output\":\"" << output_name << "\"";

        if (upload_path.size()) {
            outgoing.headers["Content-Type"] = png ? "image/png" : "image/jpeg";
            send_file(
                out,
                outgoing,
                basedir + output_name);
            return;
        }
    }
            
    sout << "}" << endl;

    write_http_response(out, outgoing, sout.str());
}


void image_server::on_post_resize(std::ostream& out, const incoming_things& incoming, outgoing_things& outgoing) {
    
    auto filepath = incoming.queries["filepath"];
    auto upload_path = incoming.queries["files/file/path"];
    if (upload_path.size())
        filepath = upload_path;

    matrix<rgb_pixel> img;

    auto basedir = get_dirname(filepath);
    string converted_name;
    
    try {
        load_image(img,  filepath);

        if (img.nr() > 1000 && img.nc() > 1000) {
            double scale = ((double)1000) / img.nr();
            resize_image(scale, img);
            
            auto output_file = random_string() + (png ? ".png" : ".jpg");
            if (!png) save_jpeg(img, basedir + output_file);
            else save_png(img, basedir + output_file);

            converted_name = output_file;
        }
    } catch(exception& e) {
        cout << e.what() << endl;
    }
    
    if (!converted_name.size()) {
        outgoing.headers["content-type"] = "application/json";
        write_http_response(out, outgoing, "{}");
        return;
    }

    if (!upload_path.size()) {
        ostringstream sout;
        sout << "{";
        sout << "\"path\":\"" << converted_name << "\"";
        sout << "}";
        outgoing.headers["content-type"] = "application/json";
        write_http_response(out, outgoing, sout.str());
    } else {
        outgoing.headers["Content-Type"] = png ? "image/png" : "image/jpeg";
        send_file(
            out,
            outgoing,
            basedir + converted_name);
    }
}

void image_server::install_routes() {
    get("/", [&] (std::ostream& out, const incoming_things& incoming, outgoing_things& outgoing, const std::vector<string>&) {
        ostringstream sout;
    
        // We are going to send back a page that contains an HTML form with two text input fields.
            // One field called name.  The HTML form uses the post method but could also use the get
            // method (just change method='post' to method='get').
            sout << " <html> <body> "
                << "<form action='/detect' method='post'> "
                << "File: <input name='filepath' type='text'><br>  "
                << "Spec: <input name='spec' type='text'><br>  "
                << "<input type='submit'> "
                << " </form>"; 

            // Write out some of the inputs to this request so that they show up on the
            // resulting web page.
            sout << "<br>  path = "         << incoming.path << endl;
            sout << "<br>  request_type = " << incoming.request_type << endl;
            sout << "<br>  content_type = " << incoming.content_type << endl;
            sout << "<br>  protocol = "     << incoming.protocol << endl;
            sout << "<br>  foreign_ip = "   << incoming.foreign_ip << endl;
            sout << "<br>  foreign_port = " << incoming.foreign_port << endl;
            sout << "<br>  local_ip = "     << incoming.local_ip << endl;
            sout << "<br>  local_port = "   << incoming.local_port << endl;


            sout << "<br/><br/>";

            sout << "<h2>HTTP Headers the web browser sent to the server</h2>";
            // Echo out all the HTTP headers we received from the client web browser
            for ( key_value_map_ci::const_iterator ci = incoming.headers.begin(); ci != incoming.headers.end(); ++ci )
            {
                sout << "<br/>" << ci->first << ": " << ci->second << endl;
            }

        sout << "</body> </html>";

        write_http_response(out, outgoing, sout.str());
    });

    post("/detect", [&] (std::ostream& out, const incoming_things& incoming, outgoing_things& outgoing, const std::vector<string>&) {
        on_post_detect(out, incoming, outgoing);
    });

    post("/resize", [&] (std::ostream& out, const incoming_things& incoming, outgoing_things& outgoing, const std::vector<string>&) {
        on_post_resize(out, incoming, outgoing);
    });

    get("/detect", [&] (std::ostream& out, const incoming_things& incoming, outgoing_things& outgoing, const std::vector<string>&) {
        ostringstream sout;
    
        sout << " <html> <body> "
                << "<form action='/detect' method='post' enctype='multipart/form-data'> "
                << "File: <input name='file' type='file'><br>  "
                << "Spec: <input name='spec' type='text'><br>  "
                << "<input type='submit'> "
                << " </form>"; 

        sout << "<br/><br/>";
        sout << "</body> </html>";

        write_http_response(out, outgoing, sout.str());
    });

    get("/resize", [&] (std::ostream& out, const incoming_things& incoming, outgoing_things& outgoing, const std::vector<string>&) {
        ostringstream sout;
    
        sout << " <html> <body> "
                << "<form action='/resize' method='post' enctype='multipart/form-data'> "
                << "File: <input name='file' type='file'><br>  "
                << "<input type='submit'> "
                << " </form>"; 

        sout << "<br/><br/>";
        sout << "</body> </html>";

        write_http_response(out, outgoing, sout.str());
    });

    static_serve("/images", "tmp");
}

// ----------------------------------------------------------------------------------------

int main(int argc, const char* argv[])
{
    string cfgfile = "config.txt";
    std::string runtime_dir = "";
    if (argc > 1) {
        cfgfile = argv[1];
        runtime_dir = get_dirname(cfgfile);
    }
    try
    {
        config_reader cr(cfgfile);

        int port = get_option(cr,"port", 5000); 
        int face_marks = get_option(cr,"face_marks", 5); 
        int yolo = get_option(cr,"yolo", 1); 
        int mmod = get_option(cr,"mmod", 0); 
        int upsize = get_option(cr,"upsize", 800); 
        thres = get_option(cr,"thres", 0.6); 
        box = get_option(cr,"box", 3); 
        mark = get_option(cr,"mark", 3);
        png = get_option(cr,"png", 0);

        // create an instance of our web server
        image_server our_web_server(runtime_dir, face_marks, mmod, upsize, yolo);
        our_web_server.install_routes();

        // our_web_server.set_max_connections(1);

        // make it listen on port 5000
        our_web_server.set_listening_port(port);
        // Tell the server to begin accepting connections.
        our_web_server.start();
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
}
