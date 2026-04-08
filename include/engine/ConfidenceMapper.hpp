#pragma once

namespace asdun {

class ConfidenceMapper {
 public:
  /**
   * 设距离为 $d$，阈值为 $d_t$，温度为 $\tau$，映射函数为：
   * $$
   * p(d)=\frac{1}{1+\exp\left(\frac{d-d_t}{\tau}\right)},\quad
   * \text{confidence}(d)=100\cdot p(d)
   * $$
   */
  static float distanceToPercent(float distance, float threshold, float tau);
};

}  // namespace asdun
