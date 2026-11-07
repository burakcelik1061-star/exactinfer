// CUDA olmadan derlendiğinde GPU arka ucu yerine geçer.
#include <stdexcept>

#include "model.h"

namespace el {

bool cuda_available(std::string* why) {
    if (why) *why = "bu surum CUDA olmadan derlendi";
    return false;
}

std::unique_ptr<Backend> make_cuda_backend(const Gguf&, int) {
    throw std::runtime_error("bu surum CUDA olmadan derlendi");
}

}  // namespace el
