#include "engine/ConfidenceMapper.hpp"

#include <cmath>

namespace asdun {

float ConfidenceMapper::distanceToPercent(float distance, float threshold, float tau) {
  const float safe_tau = (tau > 1e-6F) ? tau : 1e-3F;
  const float z = (distance - threshold) / safe_tau;
  const float p = 1.0F / (1.0F + std::exp(z));
  return 100.0F * p;
}

}  // namespace asdun

