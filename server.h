#pragma once

#include <dlib/server.h>
#include <functional>

using namespace dlib;
using namespace std;

class web_server : public server_http
{
private:
    using route_function = std::function<void(std::ostream& out, const incoming_things&, outgoing_things&, const std::vector<string>&)>;
    std::vector<std::pair<string, route_function>> post_functions;
    std::vector<std::pair<string, route_function>> get_functions;
    std::vector<std::pair<string, string>> static_paths;

protected:
  string upload_dir;

    void read_body (
        std::istream& in,
        incoming_things& incoming
    );

public:
    web_server();

    const std::string on_request ( 
        const incoming_things& incoming,
        outgoing_things& outgoing
    ) { return ""; }

    virtual void on_connect (
            std::istream& in,
            std::ostream& out,
            const std::string& foreign_ip,
            const std::string& local_ip,
            unsigned short foreign_port,
            unsigned short local_port,
            uint64
        );

    void write_http_response_head (
        std::ostream& out,
        outgoing_things outgoing,
        long content_size
    );

    void send_file(
        std::ostream& out,
        outgoing_things outgoing,
        const string& path);

    void post(const string& path, route_function);
    void get(const string& path, route_function);
    void static_serve(const string& path, const string& store_path);
};

