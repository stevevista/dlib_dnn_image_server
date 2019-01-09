#pragma once

#include <vector>
#include <string>

using namespace std;

string random_string(const size_t length = 32);
std::vector<string> split_path(const string& fullpath);
string get_dirname(const string& fullpath);
std::vector<float> split_float_array(const string& name);
