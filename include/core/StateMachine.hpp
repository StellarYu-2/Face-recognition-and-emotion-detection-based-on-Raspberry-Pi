#pragma once

#include <mutex>

#include "core/Types.hpp"

namespace asdun {

class StateMachine {
 public:
  explicit StateMachine(AppState initial = AppState::MainMenu);

  AppState getState() const;
  void setState(AppState state);

 private:
  mutable std::mutex mutex_{};
  AppState state_{AppState::MainMenu};
};

}  // namespace asdun

