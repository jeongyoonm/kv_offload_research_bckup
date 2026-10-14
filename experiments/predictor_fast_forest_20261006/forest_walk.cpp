// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// CPU-only evaluator of arrays already bounded and validated by LatencySurface.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

extern "C" int predictor_forest_abi() noexcept { return 1; }

extern "C" double predictor_forest_walk(
    size_t trees, size_t width, size_t features, const int32_t* left,
    const int32_t* right, const int32_t* feature, const double* threshold,
    const double* value, const float* input) noexcept {
  double sum = 0;
  if (!trees || !width) return std::numeric_limits<double>::quiet_NaN();
  for (size_t tree = 0; tree < trees; ++tree) {
    size_t node = 0;
    bool leaf = false;
    for (size_t depth = 0; depth <= 32; ++depth) {
      if (node >= width) return std::numeric_limits<double>::quiet_NaN();
      const size_t offset = tree * width + node;
      if (left[offset] == -1) {
        sum += value[offset];
        leaf = true;
        break;
      }
      if (feature[offset] < 0 || size_t(feature[offset]) >= features)
        return std::numeric_limits<double>::quiet_NaN();
      // Match sklearn's float32 inputs and float64 thresholds without rounding
      // thresholds or enabling fast-math.
      node = input[feature[offset]] > threshold[offset] ? right[offset]
                                                        : left[offset];
    }
    if (!leaf) return std::numeric_limits<double>::quiet_NaN();
  }
  return sum / double(trees);
}
