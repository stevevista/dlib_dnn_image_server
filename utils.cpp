#include "utils.h"
#include <random>
#include <sstream>

string random_string(const size_t length) {
  
  const std::string chars("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

  std::default_random_engine rng(std::random_device{}());

  std::uniform_int_distribution<> dist(0, static_cast<int>(chars.size())-1);

  std::string result(length, '0');
  for (auto& chr : result) {
      chr = chars[dist(rng)];
  }
  return result;
}

std::vector<string> split_path(const std::string& fullpath) {
    std::string dirname;
    std::string basename = fullpath;
    std::string extname;

    auto div = fullpath.rfind('/');
    if (div == std::string::npos) {
        div = fullpath.rfind('\\');
    }
    if (div != std::string::npos) {
        dirname = fullpath.substr(0, div+1);
        basename = fullpath.substr(div+1);
    }

    div = basename.rfind('.');
    if (div != std::string::npos) {
      extname = basename.substr(div);
      basename = basename.substr(0, div);
    }

    return {dirname, basename, extname};
}

string get_dirname(const string& fullpath) {
  auto parts = split_path(fullpath);
  return parts[0];
}

std::vector<float> split_float_array(const string& name) {
  std::stringstream ss(name);
    std::string item;
    std::vector<float> names;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) break;
        names.push_back(atof(item.c_str()));
    }

  return names;
}
