#include <dlfcn.h>
#include <filesystem>
#include <string>
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace parallax_ext {

std::string get_type_string(mx::Dtype t);

std::string current_binary_dir();

} // namespace parallax_ext
