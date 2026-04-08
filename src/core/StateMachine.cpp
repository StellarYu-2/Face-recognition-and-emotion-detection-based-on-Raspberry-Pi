#include "core/StateMachine.hpp"

namespace asdun {

StateMachine::StateMachine(AppState initial) : state_(initial) {}

AppState StateMachine::getState() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

void StateMachine::setState(AppState state) {
  std::lock_guard<std::mutex> lock(mutex_);
  state_ = state;
}

}  // namespace asdun

