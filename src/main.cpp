#include <exception>
#include <iostream>

#include "core/App.hpp"

int main() {
  try {
    asdun::App app("./config/app.yaml");
    return app.run();
  } catch (const std::exception& ex) {
    std::cerr << "[FATAL] " << ex.what() << std::endl;
    return 1;
  }
}

