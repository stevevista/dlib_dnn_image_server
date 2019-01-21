#include "server.h"
#include "multipart.h"
#include <sstream>
#include "utils.h"

static std::vector<string> split_route(const string& name) {

    std::stringstream ss(name);
    std::string item;
    std::vector<std::string> names;
    while (std::getline(ss, item, '/')) {
        names.push_back(item);
    }

    if (names.size() && names.back().empty())
      names.erase(names.end()-1); 

    return names;
}

static bool path_match(const string& pattern, const string& path, std::vector<string>& matched) {

  auto path_parts = split_route(path);
  auto pattern_parts = split_route(pattern);

  size_t j = 0;
  for(size_t i = 0; i < pattern_parts.size(); i++) {
    auto& filter = pattern_parts[i];
    auto optional = !filter.empty() && filter[0] == '?';
    auto wildcard = !filter.empty() && filter[0] == '*';
    auto catcher = !filter.empty() && filter[0] == '+';
    
    if (wildcard) {
      auto next = (i+1) < pattern_parts.size() ? pattern_parts[i+1] : "";
      string val;
      for (; j < path_parts.size(); j++) {
        if (next == path_parts[j]) {
          break;
        }
        if (val.size()) val += "/";
        val += path_parts[j];
      }
      matched.push_back(val);
    } else {
      if (j == path_parts.size()) {
        if (optional) {
          continue;
        }

        matched.clear();
        return false;
      }

        if (optional || catcher) {
            matched.push_back(path_parts[j++]);
        } else if (filter != path_parts[j++]) {
            matched.clear();
            return false;
        }
    }
  }

  bool ret = (j == path_parts.size());
  if (!ret) {
    matched.clear();
  }

  return ret;
}


web_server::web_server() {
  upload_dir = "tmp/";
}

void web_server::read_body (
        std::istream& in,
        incoming_things& incoming
    ) {

    // if the body hasn't already been loaded and there is data to load
    if (incoming.body.size() == 0 &&
            incoming.headers.count("Content-Length") != 0)
        {
            const unsigned long content_length = string_cast<unsigned long>(incoming.headers["Content-Length"]);

            string boundary;
            if (incoming.content_type.find("multipart/form-data;") == 0) {
                auto pos = incoming.content_type.find("boundary=");
                if (pos != string::npos) {
                    boundary = "--" + incoming.content_type.substr(pos+9);
                }
            } 
            
            if (boundary.size()) {
                MultipartParser parser(boundary, in, content_length);

                parser.parse([&](const string& name, const string& filename, const string& value, DataParser* data_parser) {
                    // cout << "field: " << name << ", filename: " << filename << ", value: " << value << endl;
                    if (filename.size()) {
                        if (data_parser) {
                            auto parts = split_path(filename);
                            auto storename = random_string() + parts[2];
                            auto path = upload_dir + storename;
                            std::ofstream ofs(path, ofstream::binary);
                            
                            unsigned char buff[4096 + 4];
                            while (true) {
                                auto nread = data_parser->read_at_least(buff, 4096);
                                ofs.write((const char*)buff, nread);
                                if (nread < 4096) {
                                    break;
                                }
                            }
                            ofs.close();

                            incoming.queries["files/" + name + "/name"] = filename;
                            incoming.queries["files/" + name + "/path"] = path;
                        }
                    } else {
                        incoming.queries[name] = value;
                    }
                });
            }
            else {
                incoming.body.resize(content_length);
                if (content_length > 0)
                {
                    in.read(&incoming.body[0],content_length);
                }
            }
  }
}

void web_server::write_http_response_head (
        std::ostream& out,
        outgoing_things outgoing,
        long content_size
    ) 
{
  key_value_map_ci& response_headers = outgoing.headers;

  // only send this header if the user hasn't told us to send another kind
  bool has_content_type = false, has_location = false;
        for(key_value_map_ci::const_iterator ci = response_headers.begin(); ci != response_headers.end(); ++ci )
        {
            if ( !has_content_type && strings_equal_ignore_case(ci->first , "content-type") )
            {
                has_content_type = true;
            }
            else if ( !has_location && strings_equal_ignore_case(ci->first , "location") )
            {
                has_location = true;
            }
        }

        if ( has_location )
        {
            outgoing.http_return = 302;
        }

        if ( !has_content_type )
        {
            response_headers["Content-Type"] = "text/html";
        }

  response_headers["Content-Length"] = cast_to_string(content_size);

  out << "HTTP/1.0 " << outgoing.http_return << " " << outgoing.http_return_status << "\r\n";

  // Set any new headers
  for(key_value_map_ci::const_iterator ci = response_headers.begin(); ci != response_headers.end(); ++ci )
  {
      out << ci->first << ": " << ci->second << "\r\n";
  }

  out << "\r\n";
}

void web_server::on_connect (
            std::istream& in,
            std::ostream& out,
            const std::string& foreign_ip,
            const std::string& local_ip,
            unsigned short foreign_port,
            unsigned short local_port,
            uint64
        ) {

  try {
      incoming_things incoming(foreign_ip, local_ip, foreign_port, local_port);
      outgoing_things outgoing;

      parse_http_request(in, incoming, get_max_content_length());
      read_body(in, incoming);

      cout << incoming.request_type << " " << incoming.path << " Body:" << incoming.body << endl;

      std::vector<string> matched;

      if (incoming.request_type == "POST") {
        for (const auto& rt : post_functions) {
            auto r = path_match(rt.first, incoming.path, matched);
            if (r) {
                rt.second(out, incoming, outgoing, matched);
                return;
            }
        }
      }

      if (incoming.request_type == "GET") {
        for (const auto& rt : get_functions) {
            auto r = path_match(rt.first, incoming.path, matched);
            if (r) {
                rt.second(out, incoming, outgoing, matched);
                return;
            }
        }

        // not handled, search in static route
        // search for static server
        for (const auto& p : static_paths) {
          if (incoming.path.find(p.first) == 0) {
            auto file_part = incoming.path.substr(p.first.size());
            auto filepath = p.second + file_part;
            send_file(out, outgoing, filepath);
            return;
          }
        }
      }

        // not found
        outgoing.http_return = 404;
        outgoing.http_return_status = "Not Found";
        write_http_response(out, outgoing, "Not Found");

    } catch (http_parse_error& e) {
                // dlog << LERROR << "Error processing request from: " << foreign_ip << " - " << e.what();
      write_http_response(out, e);
    } catch (std::exception& e) {
                // dlog << LERROR << "Error processing request from: " << foreign_ip << " - " << e.what();
      write_http_response(out, e);
  }
}

void web_server::post(const string& path, route_function fn) {
  post_functions.push_back(std::make_pair(path, fn));
}

void web_server::get(const string& path, route_function fn) {
  get_functions.push_back(std::make_pair(path, fn));
}

void web_server::static_serve(const string& path, const string& store_path) {
  static_paths.push_back(std::make_pair(path, store_path));
}

void web_server::send_file(
        std::ostream& out,
        outgoing_things outgoing,
        const string& filepath) {

    bool has_content_type = false;
    for(auto& ci : outgoing.headers) {
        if ( strings_equal_ignore_case(ci.first , "content-type") ) {
            has_content_type = true;
            break;
        }
    }

    if (!has_content_type) {
        outgoing.headers["Content-Type"] = "application/octet-stream";
    }

    try {
        std::ifstream ifs(filepath, std::ifstream::binary);

        auto begin = ifs.tellg();
              ifs.seekg (0, ios::end);
              auto end = ifs.tellg();
              ifs.seekg (0, ios::beg);
              auto size = end - begin;

              write_http_response_head(out, outgoing, size);
              char buffer[4096];

        while (true) {
                ifs.read(buffer, 4096);
                auto nread = ifs.gcount();
                out.write(buffer, nread);
                if (nread != 4096) {
                  break;
                }
        }

    } catch(std::exception& e) {
        // not found
        outgoing.http_return = 404;
        outgoing.http_return_status = "Not Found";
        write_http_response(out, outgoing, "Not Found");
    }
}
