#include <dlfcn.h>
#include <filesystem>
#include <string>
#include "utils.h"

namespace parallax_ext {

std::string get_type_string(mx::Dtype t) {
    if (t == mx::float32) return "float";
    if (t == mx::float16) return "half";
    if (t == mx::bfloat16) return "bfloat16_t";
    if (t == mx::uint8) return "uchar";
    throw std::runtime_error("Unsupported dtype");
}

std::string current_binary_dir()
{
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

} // namespace parallax_ext
